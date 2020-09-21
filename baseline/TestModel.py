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


from DataLoad import DataLoadDf
from DatasetDcase2019Task4 import DatasetDcase2019Task4
from evaluation_measures import audio_tagging_results, get_f_measure_by_class, compute_strong_metrics, get_predictions
from utils.utils import ManyHotEncoder, to_cuda_if_available, get_transforms
from utils.Logger import LOG
from utils.Scaler import Scaler
from models.CRNN import CRNN
from models.SACNN import CNN
from models.SACNN import SACNN
import config as cfg

from pdb import set_trace as pause


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
    LOG.info("Model loaded at epoch: {}".format(state["epoch"]))
    pooling_time_ratio = state["pooling_time_ratio"]

    model_nn.load(parameters=state["model"]["state_dict"])
    scaler = Scaler()
    scaler.load_state_dict(state["scaler"])
    classes = cfg.classes
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])

    model_nn = model_nn.eval()
    [model_nn] = to_cuda_if_available([model_nn])
    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)

    LOG.info(reference_tsv_path)
    df, ref_wav_dir = dataset.initialize_and_get_df(reference_tsv_path, reduced_number_of_data)
    strong_dataload = DataLoadDf(df, ref_wav_dir, feature_retrieval_func, many_hot_encoder.encode_strong_df,
                                 transform=transforms_valid)

    predictions, att_predictions = get_predictions(model_nn, strong_dataload, many_hot_encoder.decode_strong, pooling_time_ratio,
                                  save_predictions=store_predicitions_fname)
    
    compute_strong_metrics(att_predictions, df)

    weak_dataload = DataLoadDf(df, ref_wav_dir, feature_retrieval_func, many_hot_encoder.encode_weak,
                               transform=transforms_valid)
    weak_metric = get_f_measure_by_class(model_nn, len(classes), DataLoader(weak_dataload, batch_size=cfg.batch_size))
    LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
    LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

    # Just an example of how to get the weak predictions from dataframes.
    # print(audio_tagging_results(df, predictions))


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
    else:
        raise NotImplementedError("Model not found, please use of one of implemented models")

    reduced_number_of_data = f_args.subpart_data
    model_path = f_args.model_path
    expe_state = torch.load(model_path, map_location="gpu")
    
    test_model(model, expe_state, cfg.eval2018, reduced_number_of_data)
    test_model(model, expe_state, cfg.validation, reduced_number_of_data, "validation2019_predictions.tsv")

    test_model(model, expe_state, cfg.eval_desed, reduced_number_of_data, "eval2019_predictions.tsv")



