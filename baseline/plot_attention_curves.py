# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################
import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn 

from DataLoad import DataLoadDf
from DatasetDcase2019Task4 import DatasetDcase2019Task4
from evaluation_measures import audio_tagging_results, get_f_measure_by_class, compute_strong_metrics, get_predictions
from utils.utils import ManyHotEncoder, to_cuda_if_available, get_transforms, filter_subclass_df
from utils.Logger import LOG
from utils.Scaler import Scaler
from models.CRNN import CRNN
from models.SACNN import CNN
from models.SACNN import SACNN
from models.CNNTR import CNNTransformer, FullTransformer

import config as cfg
import matplotlib.pyplot as plt
from pdb import set_trace as pause


def plot_strong_output(ax,target,pred):
    
    t = range(target.shape[0])
    if type(ax) is not np.ndarray:
        ax = np.array([ax])
        
    for i, a in enumerate(ax):
        a.plot(t,target[:,i])
        a.axis('off')
        a.fill_between(t,target[:,i],0)
        a.plot(t,pred[:,i],'r')
    #seaborn.despine(left=True, bottom=True, right=True) 


def test_model(model, state, reference_tsv_path, reduced_number_of_data=None, store_predicitions_fname=None):
    dataset = DatasetDcase2019Task4(os.path.join(cfg.workspace),
                                    base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                                    save_log_feature=False)
    
    if not cfg.online_feature_extraction:
        feature_retrieval_func = dataset.get_feature_file
    else:
        feature_retrieval_func = None  #if set to None will extract features on-the-fly 
                                    #directly from wav files 
                                  
    model_kwargs = state["model"]["kwargs"]
    model_nn = model(**model_kwargs)
    model_nn.load(parameters=state["model"]["state_dict"])
    
    pooling_time_ratio = state["pooling_time_ratio"]

    model_nn.load(parameters=state["model"]["state_dict"])
    scaler = Scaler()
    scaler.load_state_dict(state["scaler"])
    classes = cfg.classes
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])

    model_nn = model_nn.eval()
    [model_nn] = to_cuda_if_available([model_nn])
    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)
    
    
    df, ref_wav_dir = dataset.initialize_and_get_df(reference_tsv_path, reduced_number_of_data)
    df = filter_subclass_df(df, cfg.classes)
    
    df_frames = df.copy()
    df_frames.onset = df_frames.onset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    df_frames.offset = df_frames.offset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    #pause()
    strong_dataload = DataLoadDf(df_frames, ref_wav_dir, feature_retrieval_func, many_hot_encoder.encode_strong_df,
                                 transform=transforms_valid,return_indexes=True)

    #weak_dataload = DataLoadDf(df_frames, ref_wav_dir, feature_retrieval_func, many_hot_encoder.encode_weak,
    #                           transform=transforms_valid)

    for i, (batch_sample, index) in tqdm(enumerate(strong_dataload),total = len(strong_dataload)):
        batch_input = batch_sample[0]
        target = batch_sample[1]
        
        [batch_input, target] = to_cuda_if_available([batch_input, target])
    
        if len(batch_input.size())==3:
            batch_input = batch_input.unsqueeze(-1).unsqueeze(0)        
        strong_pred, weak_pred, att_map = model_nn(batch_input)
        if len(target.shape)==2:
            target = target.unsqueeze(0)

        target = target.unsqueeze(1).repeat(1,cfg.n_perturb+1,1,1).view(-1,target.shape[1],target.shape[2])


        
        # fig1 = plt.figure()
        # for b in range(att_map.size()[0]):
        #     for k in range(att_map.size()[-1]):
        #         plt.plot(att_map[b,:,:,k].cpu().detach().numpy().T)
        #         fig1.canvas.draw()
        #         fig1.canvas.flush_events()
        #         plt.waitforbuttonpress()
                
        #         plt.clf()
        # plt.close()        
        #------------plotting
        # fig = plt.figure()
        # plt.subplot(2,1,1);
        # plt.imshow(batch_input.squeeze()[::8,:].cpu().T,aspect = "auto");
        # plt.subplot(2,1,2);
        # plt.imshow(target.cpu().T,aspect = "auto");
        
        

        for j in range(att_map.shape[0]): 
            fig, ax = plt.subplots(cfg.cnn_transformer_kwargs['nclass'], 1, sharex=True)
            idx = int(j*att_map.shape[1])
            
            plot_strong_output(ax,target[idx,:,:].cpu().numpy()
                ,att_map[j,0,:,:].cpu().detach().numpy())
            
            #fig.canvas.draw()
            plt.show(block=False)
            fig.canvas.flush_events()
            plt.waitforbuttonpress()
            plt.clf()
        
            plt.close()        

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")
    parser.add_argument("-m", '--model_path', type=str, default=None, dest="model_path",
                        help="Path of the model to be resume or to get validation results from.")
    parser.add_argument("-p", '--save_predictions_fname', type=str, default=None, dest="save_predictions_fname",
                        help="Path for the predictions to be saved, if not set, not save them")

    parser.add_argument("-n", '--model', type=str, default='crnn', dest="model_name",
                        help="the model type to be loaded:  CRNN/CNN/...")


    

    f_args = parser.parse_args()
    if (f_args.model_name.lower() == 'crnn'):
        model = CRNN 
    elif (f_args.model_name.lower() == 'cnn'):
        model = CNN
    elif (f_args.model_name.lower() == 'sacnn'):
        model = SACNN
    elif (f_args.model_name.lower() == 'cnntr'):
        model = CNNTransformer
    elif (f_args.model_name.lower() == 'fulltr'):
        model = FullTransformer
    else:
        raise NotImplementedError("Model not found, please use of one of implemented models")

    reduced_number_of_data = f_args.subpart_data
    model_path = f_args.model_path
    expe_state = torch.load(model_path, map_location="cpu")

    #test_model(model, expe_state, cfg.eval2018, reduced_number_of_data)
    test_model(model, expe_state, cfg.synthetic, reduced_number_of_data, "")
    #test_model(model, expe_state, cfg.validation, reduced_number_of_data, "validation2019_predictions.tsv")

    #test_model(model, expe_state, cfg.eval_desed, reduced_number_of_data, "eval2019_predictions.tsv")



