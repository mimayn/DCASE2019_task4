# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################


import argparse
import os
import pandas as pd
import time
import numpy as np
from pdb import set_trace as pause
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn

from DatasetDcase2019Task4 import DatasetDcase2019Task4
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from utils.Scaler import Scaler
from TestModel import test_model
from evaluation_measures import get_f_measure_by_class, get_predictions, audio_tagging_results, compute_strong_metrics
from models.CNNTR import FullTransformer, CNNTransformer
import config as cfg
from utils.utils import ManyHotEncoder, AverageMeterSet, create_folder, SaveBest, to_cuda_if_available, weights_init, \
    get_transforms, VisdomLinePlotter, balance_df, filter_subclass_df, get_class_weights
from utils.Logger import LOG
import losses
import seaborn



def plot_strong_output(ax,target,pred):
    
    t = range(target.shape[0])
    if type(ax) is not np.ndarray:
        ax = np.array([ax])
        
    for i, a in enumerate(ax):
        a.plot(t,target[:,i])
        #a.legend(cfg.classes[i])
        a.text(-5,.5,cfg.classes[i], fontsize=6)
        #a.axis('off')
        a.set_ylim(0,1)
        a.set_yticks([])
        a.set_xticks([])

        a.fill_between(t,target[:,i],0)
        a.plot(t,pred[:,i],'r')
    
    plt.subplots_adjust(wspace=.1, hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)

    seaborn.despine(left=True, bottom=True, right=True) 

def train(train_loader, model, optimizer, epoch, weak_mask=None, strong_mask=None,figs_dir=''):
    class_criterion = nn.BCELoss()
    [class_criterion] = to_cuda_if_available([class_criterion])

    meters = AverageMeterSet()
    meters.update('lr', optimizer.param_groups[0]['lr'])

    running_loss_dict = {'total':0.,'asyn':0.,'tv':0.,'bin':0, 'synth':0.,'dom':0.,'aux':0}

    LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()

    create_folder(os.path.join(figs_dir,'perturbation_attentions'))
    create_folder(os.path.join(figs_dir,'prediction_curves'))
    for bs in range(cfg.batch_size):
        create_folder(os.path.join(figs_dir,'prediction_curves','example_{}'.format(bs)))
        create_folder(os.path.join(figs_dir,'perturbation_attentions','example_{}'.format(bs)))

    for i, (batch_input, target) in tqdm(enumerate(train_loader),total = len(train_loader)):
        [batch_input, target] = to_cuda_if_available([batch_input, target])
        LOG.debug(batch_input.mean())

        

        strong_pred, weak_pred, att_map = model(batch_input)
        
        target = target.unsqueeze(1).repeat(1,cfg.n_perturb+1,1,1).view(-1,target.shape[1],target.shape[2])

        #att_strong_pred = att_map[:,0,:,:].squeeze() 
        loss = 0
        if i==0 and epoch%1==0 and cfg.plot_attention_curves and (att_map.shape[1] > 1):
            for b in range(att_map.size()[0]):
                    my_dpi=96
                    fig, axes = plt.subplots(att_map.size()[-1], 1, sharex=True)
                    for k, ax in enumerate(axes):
                        ax.plot(att_map[b,:,:,k].cpu().detach().numpy().T)
                        ax.set_ylim(0,1)
                        #plt.ylim(top=1,bottom=0)
                    #fig1.canvas.draw()
                    #fig1.canvas.flush_events()
                    #plt.waitforbuttonpress()
                    folder = os.path.join(figs_dir,'perturbation_attentions','example_{}'.format(b))
                    plt.savefig(os.path.join(folder, 'batch_{}_epoch_{}.jpg'.format(i,epoch)),dpi=my_dpi)

                    #plt.clf()
            plt.close()        
        #------------plotting
        
        if cfg.plot_attention_curves:
            if i==0 and epoch%1==0:
                #pause()
                for j in range(att_map.shape[0]): 
                    my_dpi=96
                    fig, ax = plt.subplots(att_map.size()[-1], 1, sharex=True)
                    idx = int(j*att_map.shape[1])
                    plot_strong_output(ax,target[idx,:,:].cpu().numpy()
                        ,att_map[j,0,:,:].cpu().detach().numpy())
                    folder = os.path.join(figs_dir,'prediction_curves','example_{}'.format(j))
                    plt.savefig(os.path.join(folder, 'batch_{}_epoch_{}.jpg'.format(i,epoch)),dpi=my_dpi)
                    #fig.canvas.draw()
                    #fig.canvas.flush_events()
                    #plt.waitforbuttonpress()
                    #plt.clf()
                
                    plt.close()        

        
        #---------
        if weak_mask is not None:
            # Weak BCE Loss
            # Trick to not take unlabeled data
            # Todo figure out another way
            target_weak = target.max(-2)[0]
            
            weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
            if i == 1:
                LOG.debug("target: {}".format(target.mean(-2)))
                LOG.debug("Target_weak: {}".format(target_weak))
                LOG.debug(weak_class_loss)
            meters.update('Weak loss', weak_class_loss.item())
            running_loss_dict['dom'] += weak_class_loss.item()
            loss += weak_class_loss
            
            att_weak_mask = slice(weak_mask.start//(cfg.n_perturb+1),weak_mask.stop//(cfg.n_perturb+1))

            aux_weak_loss = losses.aux_weak_loss(att_map[att_weak_mask], target_weak[weak_mask],cfg.aux_lambda)
            loss += aux_weak_loss
            
            running_loss_dict['aux'] += aux_weak_loss.item()

        if strong_mask is not None:
            
            if cfg.use_synthetic_as_weak:
                target_weak = target.max(-2)[0]

                #plt.plot(weak_pred.cpu().detach().numpy(),'b',target_weak.cpu().detach().numpy(),'r', torch.mean(strong_pred,dim=1).cpu().detach().numpy(),'g');plt.show()
                weak_synth_class_loss = class_criterion(weak_pred[strong_mask], target_weak[strong_mask])
                running_loss_dict['synth'] += weak_synth_class_loss.item()
                loss += weak_synth_class_loss

                att_strong_mask = slice(strong_mask.start//(cfg.n_perturb+1),strong_mask.stop//(cfg.n_perturb+1))
                aux_weak_synth_loss = losses.aux_weak_loss(att_map[att_strong_mask], target_weak[strong_mask], cfg.aux_lambda)
                loss += aux_weak_synth_loss
                running_loss_dict['aux'] += aux_weak_synth_loss.item()


            else:
                # Strong BCE loss
                strong_class_loss = class_criterion(strong_pred[strong_mask], target[strong_mask])
                meters.update('Strong loss', strong_class_loss.item())
                if cfg.use_attention_curves:
                    #evaluation of temporal attention maps as alternative predictions
                    att_strong_class_loss = class_criterion(att_strong_pred[strong_mask], target[strong_mask])
                    meters.update('Attention Map Strong loss', att_strong_class_loss.item())

                loss += strong_class_loss
                running_loss_dict['synth'] += strong_class_loss.item()
        # asynchrony loss
        
        if att_map.shape[1] > 1:
            asyn_loss = losses.asynchrony_loss(att_map, asyn_measure='IoU', asyn_lambda=cfg.asyn_loss_lambda) 
            running_loss_dict['asyn'] += asyn_loss.item()
            meters.update('asynchrony loss', asyn_loss.item())
            loss += asyn_loss
        #total-variation loss
        
        tv_loss = losses.TV_loss(att_map, tv_lambda=cfg.tv_loss_lambda)
        running_loss_dict['tv'] += tv_loss.item()

        # Binarization loss
        bin_loss = losses.binarization_loss(att_map, bin_lambda=cfg.bin_loss_lambda)
        running_loss_dict['bin'] += bin_loss.item()

        meters.update('Total Variation loss', tv_loss.item())
        meters.update('Binarization loss', bin_loss.item())

        loss += tv_loss + bin_loss
        
        running_loss_dict['total'] += loss.item()

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    epoch_time = time.time() - start

    LOG.info(
        'Epoch: {}\t'
        'Time {:.2f}\t'
        '{meters}'.format(
            epoch, epoch_time, meters=meters))
    running_loss_dict.update((x,y/len(train_loader)) for x,y in running_loss_dict.items())
    return running_loss_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")
    f_args = parser.parse_args()

    reduced_number_of_data = f_args.subpart_data
    
    use_weak = cfg.use_weak
    use_synthetic = cfg.use_synthetic
    
    if cfg.visualize_performance:
        global plotter_loss, plotter_metric
        loss_plotter = VisdomLinePlotter(env_name='Loss evolution plots')
        metric_plotter = VisdomLinePlotter(env_name='Overall metric evolution')
    
    # ####################################
    # Model
    # ####################################
    
    model_kwargs = cfg.cnn_transformer_kwargs
    #pause()
    model = CNNTransformer(**model_kwargs)
        
    #model.apply(weights_init)
    pooling_time_ratio = cfg.pooling_time_ratio

    LOG.info(model)

    ################################


    if use_weak and use_synthetic:
        add_dir_path = "_synthetic_and_weak"
         
    elif use_weak:
        add_dir_path = "_weak_only"
        
    elif use_synthetic:
        add_dir_path = "_synthetic_only"    

    if cfg.add_perturbations:
        perturb_type = cfg.perturb_type
        add_dir_path += "_{}_pertubations".format(cfg.n_perturb) 
    else:
        perturb_type = None

    if cfg.asyn_loss_lambda != 0:
        add_dir_path += "asyn_{}_".format(cfg.asyn_loss_lambda)            
    if cfg.bin_loss_lambda != 0:
        add_dir_path += "bin_{}_".format(cfg.bin_loss_lambda) 
    if cfg.aux_lambda != 0:
        add_dir_path += "aux_{}_".format(cfg.aux_lambda) 
    if cfg.tv_loss_lambda != 0:
        add_dir_path += "tv_{}_".format(cfg.tv_loss_lambda)         

    #store_dir = os.path.join("/save/2016022/mabdol01/ICASSP21/stored_data", model.__class__.__name__ + add_dir_path)       
    store_dir = os.path.join("stored_data", model.__class__.__name__ + add_dir_path)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    saved_figs_dir = os.path.join(store_dir, "figures")
    create_folder(store_dir)
    create_folder(saved_model_dir)
    create_folder(saved_pred_dir)
    create_folder(saved_figs_dir)

    logger_path = os.path.join(store_dir,model.__class__.__name__+".log")    
    #LOG = create_logger("SACNN", )    
    LOG.info("Simple {}".format(model.__class__.__name__))
    LOG.info("subpart_data = {}".format(reduced_number_of_data))
   
    LOG.info("Dataset compositions : weak data:{}  Synthetic_data:{}".format(use_weak, use_synthetic))

    # ##############
    # DATA
    # ##############
    
    dataset = DatasetDcase2019Task4(os.path.join(cfg.workspace),
                                    base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                                    save_log_feature=False)
    if not cfg.online_feature_extraction:
        feature_retrieval_func = dataset.get_feature_file
    else:
        feature_retrieval_func = None  #if set to None will extract features on-the-fly 
                                    #directly from wav files 
    
    weak_df, weak_wav_dir = dataset.initialize_and_get_df(cfg.weak, reduced_number_of_data)
    weak_df = filter_subclass_df(weak_df, cfg.classes)

    synthetic_df, synthetic_wav_dir = dataset.initialize_and_get_df(cfg.synthetic, reduced_number_of_data, download=False)
    synthetic_df = filter_subclass_df(synthetic_df, cfg.classes)

    validation_df, validation_wav_dir = dataset.initialize_and_get_df(cfg.validation, reduced_number_of_data)
    validation_df = filter_subclass_df(validation_df, cfg.classes)
   
    weak_weights, weak_dict = get_class_weights(weak_df)
    synth_weights, synth_dict = get_class_weights(synthetic_df)


# ------------------------------------ create transformations -------------------------
    #dataset.extract_transformations_from_meta(weak_df, cfg.weak, cfg.n_transforms,reduced_number_of_data)
    
    #dataset.extract_transformations_from_meta(synthetic_df, cfg.synthetic, cfg.n_transforms, reduced_number_of_data)
    
# ------------------------------------------------------------------------------------------------------
   
    #classes = DatasetDcase2019Task4.get_classes([weak_df, validation_df, synthetic_df])
    #use the filtered subclass list from the config file insead

    # Be careful, frames is max_frames // pooling_time_ratio because max_pooling is applied on time axis in the model
    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio)

    #transforms = get_transforms(cfg.max_frames,augment_type='random_mask')
    transforms = get_transforms(cfg.max_frames)

    # Divide weak in train and valid
    train_weak_df = weak_df.sample(frac=0.8, random_state=26)
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)
    LOG.debug(valid_weak_df.event_labels.value_counts())
    #train_weak_data = DataLoadDf(train_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
    #                             transform=transforms)
    train_weak_data = DataLoadDf(train_weak_df, weak_wav_dir, feature_retrieval_func, many_hot_encoder.encode_strong_df,
                                 transform=transforms)
    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)


    # --------------------------------
    #balance the training data train_weak_df and train_synth_df
    train_synth_df = balance_df(train_synth_df, cfg.classes)
    train_weak_df = balance_df(train_weak_df, cfg.classes)
    # -------------------------------------------------
    
    
    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df_frames = train_synth_df.copy()
    train_synth_df_frames.onset = train_synth_df_frames.onset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    train_synth_df_frames.offset = train_synth_df_frames.offset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    LOG.debug(valid_synth_df.event_label.value_counts())
    LOG.debug(valid_synth_df)
    


    train_synth_data = DataLoadDf(train_synth_df_frames, synthetic_wav_dir, feature_retrieval_func,
                                  many_hot_encoder.encode_strong_df,
                                  transform=transforms)

    if use_weak and use_synthetic:
        list_datasets = [train_weak_data, train_synth_data]
        training_data = ConcatDataset(list_datasets)
    elif use_weak:
        list_datasets = [train_weak_data]
        training_data = train_weak_data
    elif use_synthetic:
        list_datasets = [train_synth_data]
        training_data = train_synth_data
        
    scaler = Scaler()
    scaler.calculate_scaler(training_data)
    LOG.debug(scaler.mean_)

    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)
    # Validation dataset is only used to get an idea of wha could be results on evaluation dataset
    validation_dataset = DataLoadDf(validation_df, validation_wav_dir, feature_retrieval_func, many_hot_encoder.encode_strong_df,
                                    transform=transforms_valid)

    
    transforms = get_transforms(cfg.max_frames, 
                    perturb_type = perturb_type, 
                    n_perturb = cfg.n_perturb, scaler=scaler, frac=cfg.frac)


    train_synth_data.set_transform(transforms)
    train_weak_data.set_transform(transforms)
    
    
    
    if use_weak and use_synthetic:
        
        concat_dataset = ConcatDataset([train_weak_data, train_synth_data])
        # Taking as much data from synthetic than strong.
        sampler = MultiStreamBatchSampler(concat_dataset,
                                          batch_sizes=[cfg.batch_size // 2, cfg.batch_size // 2])
        training_data = DataLoader(concat_dataset, batch_sampler=sampler)
        valid_weak_data = DataLoadDf(valid_weak_df, weak_wav_dir, feature_retrieval_func, many_hot_encoder.encode_strong_df,
                                     transform=transforms_valid)
        weak_mask = slice(0,(cfg.batch_size // 2)*(cfg.n_perturb+1) )
        strong_mask = slice((cfg.batch_size // 2)*(cfg.n_perturb+1), cfg.batch_size*(cfg.n_perturb+1))
    elif use_synthetic:
        training_data = DataLoader(train_synth_data, batch_size=cfg.batch_size, drop_last = True)
        strong_mask = slice(0,cfg.batch_size*(cfg.n_perturb+1))  # Not masking
        weak_mask = None
    elif use_weak:
        training_data = DataLoader(train_weak_data, batch_size=cfg.batch_size, drop_last = True)
        valid_weak_data = DataLoadDf(valid_weak_df, weak_wav_dir, feature_retrieval_func, many_hot_encoder.encode_strong_df,
                                     transform=transforms_valid)
        strong_mask = None  
        weak_mask = slice(0,cfg.batch_size*(cfg.n_perturb+1))  

    valid_synth_data = DataLoadDf(valid_synth_df, synthetic_wav_dir, feature_retrieval_func, many_hot_encoder.encode_strong_df,
                                  transform=transforms_valid)

    # ##############
    # Train
    # ##############
    #optim_kwargs = {"lr": 0.001, 'momentum':0.9}
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), **optim_kwargs)

    optim_kwargs = {"lr": cfg.lr, "betas": (0.9, 0.999), 'eps':1e-3}
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **optim_kwargs)
    

    LOG.info(optimizer)
    bce_loss = nn.BCELoss()

    state = {
        'model': {"name": model.__class__.__name__,
                  'args': '',
                  "kwargs": model_kwargs,
                  'state_dict': model.state_dict()},
        'optimizer': {"name": optimizer.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': optimizer.state_dict()},
        "pooling_time_ratio": pooling_time_ratio,
        'scaler': scaler.state_dict(),
        "many_hot_encoder": many_hot_encoder.state_dict()
    }

    save_best_cb = SaveBest("sup")

    # Eval 2018
    eval_2018_df, _ = dataset.initialize_and_get_df(cfg.eval2018, reduced_number_of_data)
    eval_2018_df = filter_subclass_df(eval_2018_df, cfg.classes)

    eval_2018 = DataLoadDf(eval_2018_df, validation_wav_dir, feature_retrieval_func, many_hot_encoder.encode_strong_df,
                           transform=transforms_valid)

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)


    [model] = to_cuda_if_available([model])
    for epoch in tqdm(range(cfg.n_epoch)):
        model = model.train()

        loss_dict = train(training_data, model, optimizer, epoch, weak_mask, strong_mask, saved_figs_dir)
            

        if cfg.visualize_performance:
            for k in loss_dict.keys():
                loss_plotter.plot('loss', k, 'Training stats', epoch, loss_dict[k])


        model = model.eval()
        LOG.info("Training synthetic metric:")
        
        if use_synthetic:
            train_predictions,train_att_predictions = get_predictions(model, train_synth_data, many_hot_encoder.decode_strong, pooling_time_ratio,
                                                save_predictions=None)
            
            #train metrics using networks final explicit predictions
            LOG.info("Strong metric on training synthetic results network final output")
            train_metric = compute_strong_metrics(train_predictions, train_synth_df)
            if cfg.visualize_performance:
                metric_plotter.plot('F-1','train-synthetic-strong', 'Train vs Val', 
                                    epoch, train_metric.results_class_wise_average_metrics()['f_measure']['f_measure'])

            if cfg.use_attention_curves:
                #train metrics using attention network predictions
                LOG.info("Strong metric on training synthetic results Attention network's output")
                att_train_metric = compute_strong_metrics(train_att_predictions, train_synth_df)
            
                if cfg.visualize_performance:
                    metric_plotter.plot('F-1','train-synthetic-strong(Att)', 'Train vs Val', 
                                        epoch, att_train_metric.results_class_wise_average_metrics()['f_measure']['f_measure'])

            if cfg.use_synthetic_as_weak:
                LOG.info("Training weak metric on synthetic data (used as weak labels): \n")
                train_synth_weak_metric = audio_tagging_results(train_synth_df, train_predictions)  
                if cfg.visualize_performance:
                    metric_plotter.plot('F-1','train-synth-as-weak', 'Train vs Val', epoch, np.mean(train_synth_weak_metric.values))

                print(train_synth_weak_metric)
                
        if use_weak:
            LOG.info("Training weak metric:")
            weak_metric = get_f_measure_by_class(model, len(cfg.classes),
                                                DataLoader(train_weak_data, batch_size=cfg.batch_size))
            LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
            LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))
            
            if cfg.visualize_performance:
               metric_plotter.plot('F-1','train-domestic-weak', 'Train vs Val', epoch, np.mean(weak_metric) )
            
            LOG.info("Valid weak metric:")
            weak_metric = get_f_measure_by_class(model, len(cfg.classes),
                                                 DataLoader(valid_weak_data, batch_size=cfg.batch_size))

            if cfg.visualize_performance:
                metric_plotter.plot('F-1','val-domestic-weak', 'Train vs Val', epoch, np.mean(weak_metric) )

            LOG.info(
                "Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
            LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

        
        LOG.info("Valid synthetic metric:")

        predictions, att_predictions = get_predictions(model, valid_synth_data, many_hot_encoder.decode_strong, pooling_time_ratio)

        LOG.info("Strong metric on Validation synthetic -> network final output")
        valid_metric = compute_strong_metrics(predictions, valid_synth_df)
        if cfg.visualize_performance:
                metric_plotter.plot('F-1','val-synthetic-strong', 'Train vs Val', 
                                    epoch, valid_metric.results_class_wise_average_metrics()['f_measure']['f_measure'] )
        
        if cfg.use_attention_curves:
            #attention
            LOG.info("Strong metric on Validation synthetic -> Attention network's output")
            att_valid_metric = compute_strong_metrics(att_predictions, valid_synth_df)
        
            if cfg.visualize_performance:

                    metric_plotter.plot('F-1','val-synthetic-strong(Att)', 'Train vs Val', 
                                        epoch, att_valid_metric.results_class_wise_average_metrics()['f_measure']['f_measure'] )

        #audio tagging performance on validation synthetic data        
        val_synth_weak_metric = audio_tagging_results(valid_synth_df, predictions)           
        LOG.info("Audio tagging results on validation Synthetic data:\n {}".format(val_synth_weak_metric))
        if cfg.visualize_performance:
                metric_plotter.plot('F-1','val-synth-weak', 'Train vs Val', epoch, np.mean(val_synth_weak_metric.values))


                

        if torch.cuda.device_count()>1:
                state['model']['state_dict'] = model.module.state_dict()
        else:
                state['model']['state_dict'] = model.state_dict()
        
        state['optimizer']['state_dict'] = optimizer.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_metric.results()
        
        if cfg.use_attention_curves:
            state['att_valid_metric'] = att_valid_metric.results()

        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)
            print("Saving this model: {}".format(model_fname))

        if cfg.save_best:
            if cfg.use_attention_curves:
                global_valid = att_valid_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
            else:
                global_valid = valid_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
               
            #Questionable addition
            #if use_weak and use_synthetic:
            #    global_valid += np.mean(weak_metric)
            
            if save_best_cb.apply(global_valid):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)

    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        LOG.info("testing model: {}".format(model_fname))
    else:
        LOG.info("testing model of last epoch: {}".format(cfg.n_epoch))

    # ##############
    # Validation
    # ##############

    predicitons_fname = os.path.join(saved_pred_dir, "{}_validation.tsv".format(state['model']['name']))
    test_model(CNNTransformer, state, cfg.validation, reduced_number_of_data, predicitons_fname)

    # ##############
    # Evaluation
    # ##############
    #predicitons_eval2019_fname = os.path.join(saved_pred_dir, "{}_eval2019.tsv".format(state['model']['name'])
    #test_model(CNNTransformer,state, cfg.eval_desed, reduced_number_of_data, predicitons_eval2019_fname)
