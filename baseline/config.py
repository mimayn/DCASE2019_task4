import math
import os
import pandas as pd
from pdb import set_trace as pause

#dataset_root = '/save/2016022/mabdol01/ICASSP21'
#workspace=dataset_root
workspace = ".."
# Dataset Paths
weak = 'dataset/metadata/train/weak.tsv'
unlabel = 'dataset/metadata/train/unlabel_in_domain.tsv'
synthetic = 'dataset/metadata/train/synthetic.tsv'
validation = 'dataset/metadata/validation/validation.tsv'
test2018 = 'dataset/metadata/validation/test_dcase2018.tsv'
eval2018 = 'dataset/metadata/validation/eval_dcase2018.tsv'
eval_desed = "dataset/metadata/eval/public.tsv"

#perturb_type = 'add_mixtures'
#perturb_type = 'random_mask'
perturb_type = 'add_freq_masks'
n_transforms = 20

use_attention_curves = False


use_weak = True
use_synthetic = False

use_synthetic_as_weak = True

# config
# prepare_data
sample_rate = 16000#44100 #16000
n_window = 1024#2048 #1024
hop_length = 323 #511  323
n_mels = 64
max_len_seconds = 10.
max_frames = math.ceil(max_len_seconds * sample_rate / hop_length)

f_min = 0.
f_max = 8000 #22050.

lr = 0.001
initial_lr = 0.
beta1_before_rampdown = 0.9
beta1_after_rampdown = 0.5
beta2_during_rampdup = 0.99
beta2_after_rampup = 0.999
weight_decay_during_rampup = 0.99
weight_decay_after_rampup = 0.999

max_consistency_cost = 2
max_learning_rate = 0.001

visualize_performance = True
plot_attention_curves = True

tv_loss_lambda = .02*5
asyn_loss_lambda = .3
bin_loss_lambda = .1 #.25 #0.05
aux_lambda = 1

#no_scalar = True

median_window = 5

online_feature_extraction = False

# Main
num_workers = 12
batch_size = 4
n_epoch = 250
    

add_perturbations = True
n_perturb =3
frac = .25

checkpoint_epochs = 1

save_best = True

file_path = os.path.abspath(os.path.dirname(__file__))
classes = pd.read_csv(os.path.join(file_path, "..", validation), sep="\t").event_label.dropna().sort_values().unique()

#class_selection = ['Alarm_bell_ringing','Dog']
#class_selection = ['Dog']
class_selection = None

if class_selection is not None:
  classes = class_selection

# crnn_kwargs = {"n_in_channel": 1, "nclass": len(classes), "attention": True, "n_RNN_cell": 64,
#                "n_layers_RNN": 2,
#                 "activation": "glu",
#                 "dropout": 0.5,
#                "kernel_size": 3 * [3], "padding": 3 * [1], "stride": 3 * [1], "nb_filters": [64, 64, 64],
#                 "pooling": list(3 * ((2, 4),))}


# cnn_kwargs = {"n_in_channel": 1, "nclass": len(classes), "attention": False,
#                 "activation": "Relu",
#                 "dropout": 0.5, "dilation" : 4*[(1,1)],
#                "kernel_size": 4 * [(3,3)], "pad": 4 * [(1,1)], "stride": 4 * [(1,1)], "nb_filters": [64, 128, 256, 512],
#                 "pool_size": 3*[(2,2)] + [(1,1)], 'pool_type': "max"}


# sacnn_kwargs = {"n_in_channel": 1, "nclass": len(classes), "attention": False,
#                 "activation": "Relu",
#                 "dropout": 0.5, "dilation" : 4*[(1,1)],
#                "kernel_size": 4 * [(3,3)], "pad": 4 * [(1,1)], "stride": 4 * [(1,1)], "nb_filters": [1*16, 1*32, 1*64, 1*128],
#                 "pool_size": 3*[(2,2)] + [(1,1)], 'pool_type': "max"}

cnn_transformer_kwargs = {"n_in_channel": 1, "nclass": len(classes), "activation": "glu",
                "dropout": 0.3, 'd_model': 128, 'q':16*1 , 'v':16*1 , 'h':4*4, 'N': 3,
                'attention_size': None , 'pe': None, 'chunk_mode': None, 'd_ff': 512,
                "kernel_size": 3 * [3], "pad": 3 * [1], "stride": 3 * [1], "nb_filters": [64, 64, 64],
                "pool_size": list(3 * ((2, 4),))}

#Full transformer
#cnn_transformer_kwargs = {"n_in_channel": 1, "nclass": len(classes), "dropout": 0.3, 'd_model': 128, 'q':16 , 'v':16 , 'h':4, 'N': 2,
#                'attention_size': None, 'pe': 'original', 'chunk_mode': None, 'd_ff': 256}



pooling_time_ratio = 8  # 2 * 2 * 2


model_name= 'cnntr'#'transformer'#'cnntr'
