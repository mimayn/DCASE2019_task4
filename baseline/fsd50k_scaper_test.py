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


if __name__ == '__main__':

    reduced_number_of_data = None

          
    # ##############
    # DATA
    # ##############
    
    dataset = DatasetDcase2019Task4(os.path.join(cfg.workspace),
                                    base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                                    save_log_feature=False)
    weak_df, weak_wav_dir = dataset.initialize_and_get_df(cfg.weak, reduced_number_of_data)
    weak_df = filter_subclass_df(weak_df, cfg.classes)

    synthetic_df, synthetic_wav_dir = dataset.initialize_and_get_df(cfg.synthetic, reduced_number_of_data, download=False)
    synthetic_df = filter_subclass_df(synthetic_df, cfg.classes)

# ------------------------------------ create transformations -------------------------
    dataset.extract_transformations_from_meta(weak_df, cfg.weak, cfg.n_transforms,reduced_number_of_data)
    
    dataset.extract_transformations_from_meta(synthetic_df, cfg.synthetic, cfg.n_transforms, reduced_number_of_data)
    pause()

    print('Done')