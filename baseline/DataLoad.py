# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################
import bisect
import itertools

import numpy as np
import pandas as pd
import torch
import random
import librosa
import warnings
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from pdb import set_trace as pause
import torchaudio
from utils.Logger import LOG
import config as cfg
import os
torch.manual_seed(0)
random.seed(0)






class DataLoadDf(Dataset):
    """ Class derived from pytorch Dataset
    Prepare the data to be use in a batch mode

    Args:
        df: pandas.DataFrame, the dataframe containing the set infromation (filenames, labels),
            it should contain these columns :
            "filename"
            "filename", "event_labels"
            "filename", "onset", "offset", "event_label"
        get_feature_file_func: function(), function which take a filename as input and return a feature file
        encode_function: function(), function which encode labels
        transform: function(), (Default value = None), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, (Default value = False) whether or not to return indexes when use __getitem__

    Attributes:
        df: pandas.DataFrame, the dataframe containing the set infromation (filenames, labels, ...)
        get_feature_file_func: function(), function which take a filename as input and return a feature file
        encode_function: function(), function which encode labels
        transform : function(), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, whether or not to return indexes when use __getitem__
    """
    def __init__(self, df, wav_dir ,get_feature_file_func, encode_function, transform=None,
                 return_indexes=False):

        self.df = df
        self.get_feature_file_func = get_feature_file_func
        self.encode_function = encode_function
        self.transform = transform
        self.return_indexes = return_indexes
        self.filenames = df.filename.drop_duplicates()
        self.wav_dir = wav_dir
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=cfg.sample_rate ,
                n_fft=cfg.n_window, hop_length=cfg.hop_length, f_min=cfg.f_min,
                f_max=cfg.f_max,n_mels=cfg.n_mels)

    def get_features_from_wav(self, filename, wav_dir):
        target_fs = cfg.sample_rate
        filepath = os.path.join(wav_dir,filename)
        audio, sample_rate = torchaudio.load(filepath)

        if len(audio.shape) > 1:
            audio = torch.mean(audio, axis=0)
        if target_fs is not None and sample_rate != target_fs:
            audio = torchaudio.transforms.Resample(sample_rate, target_fs)(audio)
        
        sample_rate = target_fs
        melspecgram = self.melspec_transform(audio)

        return melspecgram.t()       
    
    def set_return_indexes(self, val):
        """ Set the value of self.return_indexes

        Args:
            val : bool, whether or not to return indexes when use __getitem__
        """
        self.return_indexes = val

    def __len__(self):
        """
        Returns:
            int
                Length of the object
        """
        length = len(self.filenames)
        return length


    def get_sample(self, index):
        """From an index, get the features and the labels to create a sample

        Args:
            index: int, Index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array)
                """
        #pause()
        if self.get_feature_file_func is not None:
            features = self.get_feature_file_func(self.filenames.iloc[index])
        else:
            features = self.get_features_from_wav(self.filenames.iloc[index],self.wav_dir)    
        
        #if cfg.augment:
        #    features = stack_augmentations(features,cfg.n_aug)

        #if (len(features.shape)==2):
        #    features = np.expand_dims(features,axis=2)    # adding the channel dimension to input mel-spectrogram
  




        # event_labels means weak labels, event_label means strong labels
        if "event_labels" in self.df.columns or {"onset", "offset", "event_label"}.issubset(self.df.columns):
            if "event_labels" in self.df.columns:
                label = self.df.iloc[index]["event_labels"]
                if pd.isna(label):
                    label = []
                if type(label) is str:
                    if label == "":
                        label = []
                    else:
                        label = label.split(",")
            else:
                cols = ["onset", "offset", "event_label"]
                label = self.df[self.df.filename == self.filenames.iloc[index]][cols]
                if label.empty:
                    label = []
        else:
            label = "empty"  # trick to have -1 for unlabeled data and concat them with labeled
            if "filename" not in self.df.columns:
                raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(self.df.columns))
        if index == 0:
            LOG.debug("label to encode: {}".format(label))
        if self.encode_function is not None:
            # labels are a list of string or list of list [[label, onset, offset]]
            y = self.encode_function(label)
        else:
            y = label
        sample = features, y
        
        return sample

    def __getitem__(self, index):
        """ Get a sample and transform it to be used in a model, use the transformations

        Args:
            index : int, index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array) or
            Tuple containing the features, the labels and the index (numpy.array, numpy.array, int)

        """
        sample = self.get_sample(index)
        
        if self.transform:
            sample = self.transform(sample)

        if self.return_indexes:
            sample = (sample, index)
           
        return sample

      

    def set_transform(self, transform):
        """Set the transformations used on a sample

        Args:
            transform: function(), the new transformations
        """
        self.transform = transform

    def add_transform(self, transform):
        if type(self.transform) is not Compose:
            raise TypeError("To add transform, the transform should already be a compose of transforms")
        transforms = self.transform.add_transform(transform)
        return DataLoadDf(self.df, self.get_feature_file_func, self.encode_function, transforms, self.return_indexes)


class GaussianNoise:
    """ Apply gaussian noise
        Args:
            mean: float, the mean of the gaussian distribution.
            std: float, standard deviation of the gaussian distribution.
        Attributes:
            mean: float, the mean of the gaussian distribution.
            std: float, standard deviation of the gaussian distribution.
        """

    def __init__(self, mean=0, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) is tuple:
            sample = list(sample)
        # sample must be a tuple or a list, not apply on labels
        for k in range(len(sample) - 1):
            sample[k] = sample[k] + np.abs(np.random.normal(0, 0.5 ** 2, sample[k].shape))

        return sample


class ApplyLog(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """ Apply the transformation
        Args:

        sample: tuple, a sample defined by a DataLoad class

        Returns:
            tuple
            The transformed tuple
        """
        # sample must be a tuple or a list, first parts are input, then last element is label
        if type(sample) is tuple:
            sample = list(sample)
        for i in range(len(sample) - 1):
            sample[i] = librosa.amplitude_to_db(sample[i].T).T
        return sample


def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.

    Returns:
      ndarray, Padded or truncated input sequence data.
    """

    #if (len(x.shape))==3:
    #    time_axis=1
    #else:
    #    time_axis=0    
    time_axis=0
    length = x.shape[time_axis]
    shape = x.shape
    if length < max_len:
        pad_shape = list(shape)
        pad_shape[time_axis]=max_len - length
        #pad_shape = (max_len - length,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=time_axis)
    elif length > max_len:
        if time_axis==0:
            x_new = x[0:max_len]
    else:
        x_new = x
    return x_new


class PadOrTrunc:
    """ Pad or truncate a sequence given a number of frames
    Args:
        nb_frames: int, the number of frames to match
    Attributes:
        nb_frames: int, the number of frames to match
    """

    def __init__(self, nb_frames):
        self.nb_frames = nb_frames

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) is tuple:
            sample = list(sample)
        # sample must be a tuple or a list

        for k in range(len(sample) - 1):
            sample[k] = pad_trunc_seq(sample[k], self.nb_frames)

        return sample


class AugmentGaussianNoise:
    """ Pad or truncate a sequence given a number of frames
           Args:
               mean: float, mean of the Gaussian noise to add
           Attributes:
               std: float, std of the Gaussian noise to add
           """

    def __init__(self, mean=0, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        sample, label = sample

        noise = sample + np.abs(np.random.normal(0, 0.5 ** 2, sample.shape))

        return sample, noise, label


class AddSprinkleMaskPerturbation:
    """ create a perturbation of an original T-F representation (mel-spectrogram)
        by randomly setting a fraction of the T-F values to zero. A given number of 
        independently generated number of these perturbation are stacked on top of 
        each other, concatenated with the original T-F image and returned as ouput  
           Args:
               n_perturb: number of perturbations to create
               perturb_type : kinf of perturbation, currently only "random_mask"
               frac = fraction of pixels in the input representation to be distorted
           Attributes:
               stack_perturbations: stack all the perturbed input images along the depth of 
                        the input representation and concatenate with the original input 
                        representation
                
           """
    def __init__(self, n_perturb=5, perturb_type='random_mask',frac=0.25):
        self.n_perturb = n_perturb
        self.perturb_type = perturb_type
        self.frac = frac

    
    def stack_perturbations(self, features,n_perturb=5,perturb_type='random_mask',frac=.25):

        if perturb_type == 'random_mask':

            masks = self.create_masks(features,n_perturb,frac=self.frac)
            masked_specs = (np.tile(features,(n_perturb,1,1))*masks).transpose(1,2,0)
            return np.concatenate((np.expand_dims(features,axis=2),masked_specs),axis=2)

            
    def create_masks(self, spec,num_masks=5, frac=0.25):
    
        n = spec.shape[0]*spec.shape[1]
        mask_arr = np.ones((num_masks, n))
        n_points = int(np.ceil(n*frac))
        mask_arr[:,:n_points] = 0
        [np.random.shuffle(mask_arr[i,:]) for i in range(num_masks)]
        masks = np.reshape(mask_arr,(num_masks,*spec.shape))
        return masks


    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        feat, label = sample

        sample = self.stack_perturbations(feat, self.n_perturb, self.perturb_type, self.frac)

        return sample, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    Args:
        unsqueeze_axis: int, (Default value = None) add an dimension to the axis mentioned.
            Useful to add a channel axis to use CNN.
    Attributes:
        unsqueeze_axis: int, add an dimension to the axis mentioned.
            Useful to add a channel axis to use CNN.
    """
    def __init__(self, unsqueeze_axis=None):
        self.unsqueeze_axis = unsqueeze_axis

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample : tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) is tuple:
            sample = list(sample)
        # sample must be a tuple or a list, first parts are input, then last element is label
        for i in range(len(sample)):
            sample[i] = torch.from_numpy(sample[i]).float()  # even labels (we don't loop until -1)

        for i in range(len(sample) - 1):
            if self.unsqueeze_axis is not None:
                sample[i] = sample[i].unsqueeze(self.unsqueeze_axis)

        return sample


class Normalize(object):
    """Normalize inputs
    Args:
        scaler: Scaler object, the scaler to be used to normalize the data
    Attributes:
        scaler : Scaler object, the scaler to be used to normalize the data
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) is tuple:
            sample = list(sample)
        # sample must be a tuple or a list
        for k in range(len(sample) - 1):
            sample[k] = self.scaler.normalize(sample[k])

        return sample


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms: list of ``Transform`` objects, list of transforms to compose.
        Example of transform: ToTensor()
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def add_transform(self, transform):
        t = self.transforms.copy()
        t.append(transform)
        return Compose(t)

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Args:
        datasets : sequence, list of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @property
    def cluster_indices(self):
        cluster_ind = []
        prec = 0
        for size in self.cumulative_sizes:
            cluster_ind.append(range(prec, size))
            prec = size
        return cluster_ind

    def __init__(self, datasets):
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    @property
    def df(self):
        df = self.datasets[0].df
        for dataset in self.datasets[1:]:
            df = pd.concat([df, dataset.df], axis=0, ignore_index=True, sort=False)
        return df


class Subset(DataLoadDf):
    """
    Subset of a dataset to be used when separating in multiple subsets

    Args:
        dataload_df: DataLoadDf or similar, dataset to be split
        indices: sequence, list of indices to keep in this subset
    """
    def __init__(self, dataload_df, indices):
        self.indices = indices
        self.df = dataload_df.df.loc[indices].reset_index(inplace=False, drop=True)

        super(Subset, self).__init__(self.df, dataload_df.get_feature_file_func, dataload_df.encode_function,
                                     dataload_df.transform, dataload_df.return_indexes)

    def __getitem__(self, idx):
        return super(Subset, self).__getitem__(idx)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Args:
        dataset: Dataset, dataset to be split
        lengths: sequence, lengths of splits to be produced
    """
    # if ratio > 1:
    # 	raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = np.random.permutation(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(itertools.accumulate(lengths), lengths)]


def train_valid_split(dataset, validation_amount):
    valid_length = int(validation_amount * len(dataset))
    train_length = len(dataset) - valid_length

    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])
    return train_dataset, valid_dataset


class ClusterRandomSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size=None, shuffle=True):
        super(ClusterRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        if batch_size is not None:
            assert self.data_source.batch_sizes is None, "do not declare batch size in sampler " \
                                                         "if data source already got one"
            self.batch_sizes = [batch_size for _ in self.data_source.cluster_indices]
        else:
            self.batch_sizes = self.data_source.batch_sizes
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for j, cluster_indices in enumerate(self.data_source.cluster_indices):
            batches = [
                cluster_indices[i:i + self.batch_sizes[j]] for i in range(0, len(cluster_indices), self.batch_sizes[j])
            ]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_sizes[j]]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)

            # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        return iter(lst)

    def __len__(self):
        return len(self.data_source)


class MultiStreamBatchSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_sizes, shuffle=True):
        super(MultiStreamBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_sizes = batch_sizes
        l_bs = len(batch_sizes)
        nb_dataset = len(self.data_source.cluster_indices)
        assert l_bs == nb_dataset, "batch_sizes must be the same length as the number of datasets in " \
                                   "the source {} != {}".format(l_bs, nb_dataset)
        self.shuffle = shuffle

    def __iter__(self):
        indices = self.data_source.cluster_indices
        if self.shuffle:
            for i in range(len(self.batch_sizes)):
                indices[i] = np.random.permutation(indices[i])
        iterators = []
        for i in range(len(self.batch_sizes)):
            iterators.append(grouper(indices[i], self.batch_sizes[i]))

        return (sum(subbatch_ind, ()) for subbatch_ind in zip(*iterators))

    def __len__(self):
        val = np.inf
        for i in range(len(self.batch_sizes)):
            val = min(val, len(self.data_source.cluster_indices[i]) // self.batch_sizes[i])
        return val


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n

    return zip(*args)
