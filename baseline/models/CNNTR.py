import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import reduce
from pdb import set_trace as pause
from time import time
import config as cfg
from models.CNN import CNN

from tst.encoder import Encoder
from tst.utils import generate_original_PE, generate_regular_PE
import numpy as np


class CNNTransformer(nn.Module):
    """CNN combined with blocks of Transformer encoders from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    """    

    def __init__(self, 
                n_in_channel: int,
                nclass: int, 
                activation: str,
                stride: int,
                pad: int,
                d_model: int,
                kernel_size: list,
                pool_size: list, 
                q: int,
                v: int,
                h: int,
                N: int,
                d_ff: int = 512,
                attention_size: int = None,
                dropout: float = 0.3,
                chunk_mode: bool = True,
                pe: str = None,
                nb_filters: list=[64,64,64], **kwargs):
        
        super(CNNTransformer, self).__init__()
        
        self._d_model = d_model

        self.cnn = CNN(n_in_channel=n_in_channel, stride=stride, padding =pad, kernel_size=kernel_size,
                                activation=activation, 
                                pooling=pool_size,
                                nb_filters=nb_filters,
                                 **kwargs)
        
        self.final_freq = int(cfg.n_mels/reduce(lambda x,y: (x[0]*y[0],x[1]*y[1]),pool_size)[1])
        self.cnn_out_dim = self.final_freq * nb_filters[-1]

        self._embedding = nn.Linear(self.cnn_out_dim, d_model)  
        
        
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      d_ff,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
                  
        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        #self.name = 'cnn_transformer'


        # self.mlp = nn.Sequential(
        #     nn.Linear(self._d_model, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, nclass)
        #     )
        
        self.mlp = nn.Linear(self._d_model, nclass)
               

    def load(self, filename=None, parameters=None):
        
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        


    def forward(self, x):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        in_shape = x.shape
        
        # the input tensor with the perturbation axis comes 
        # in the following dimensions:
        # (Batch x channel x Time x freq x perturb)
        if len(x.size())==5:
            x = x.permute(0,4,1,2,3)
            x = x.contiguous().view(-1,in_shape[1],in_shape[2],in_shape[3])
            perturb_dim = in_shape[-1]
        else:
            perturb_dim=1

        x = self.cnn(x)    

        
        #reshape to from [Batch x channel x Time X freq] 
        #               --> [Batch x Time x Feature]   
        #                   (feature = channel x freq)
        b,ch,t,fr = x.shape
        x = x.transpose(1,2).reshape(b,t,-1)
        
        # Embedding module
        encoding = self._embedding(x)

        #concatenate the embedding with tag token (arbitrary constant vector) 
        tag_token = 1*torch.ones(encoding.shape[0],1,encoding.shape[2]).to(encoding.device)
        encoding = torch.cat([tag_token, encoding], 1)
        
        #time-series length
        K = encoding.shape[1]
        #pause()
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)
            
            
        # Output module
        output = torch.sigmoid(self.mlp(encoding))
    

        clipwise_output = output[:,0,:]
        
        framewise_output = output[:,1:,:]
        #clipwise_output = framewise_output.max(-2)[0]
        
        bp =framewise_output.size()[0]
        t_att = framewise_output
        t_att = t_att.view(int(bp/perturb_dim),
                perturb_dim, *t_att.shape[1:])

        return framewise_output, clipwise_output, t_att  
        # (last output dummy variable, placeholder for attention, to be implemented)





class FullTransformer(nn.Module):

    def __init__(self, 
                n_in_channel: int,
                nclass: int, 
                d_model: int,
                q: int,
                v: int,
                h: int,
                N: int,
                d_ff: int = 512,
                attention_size: int = None,
                dropout: float = 0.3,
                chunk_mode: bool = True,
                pe: str = None,
                **kwargs):
        
        super(FullTransformer, self).__init__()
        
        self._d_model = d_model

        
        #self.final_freq = int(cfg.n_mels/reduce(lambda x,y: (x[0]*y[0],x[1]*y[1]),pool_size)[1])
        #self.cnn_out_dim = self.final_freq * nb_filters[-1]
        
        self._embedding = nn.Linear(cfg.n_mels, d_model)  
                
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      d_ff,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
                  
        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        #self.name = 'cnn_transformer'


        # self.mlp = nn.Sequential(
        #     nn.Linear(self._d_model, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, nclass)
        #     )
        
        self.mlp = nn.Linear(self._d_model, nclass)
               

    def load(self, filename=None, parameters=None):
        
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        


    def forward(self, x):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        in_shape = x.shape
        
        # the input tensor with the perturbation axis comes 
        # in the following dimensions:
        # (Batch x channel x Time x freq x perturb)
        if len(x.size())==5:
            x = x.permute(0,4,1,2,3)
            x = x.contiguous().view(-1,in_shape[1],in_shape[2],in_shape[3])
            perturb_dim = in_shape[-1]
        else:
            perturb_dim=1
        
               
        #reshape to from [Batch x channel x Time X freq] 
        #               --> [Batch x Time x Feature]   
        #                   (feature = channel x freq)
        b,ch,t,fr = x.shape
        x = x.transpose(1,2).reshape(b,t,-1)
        
        # Embedding module
        encoding = self._embedding(x)

        #concatenate the embedding with tag token (arbitrary constant vector) 
        tag_token = .2*torch.ones(encoding.shape[0],1,encoding.shape[2]).to(encoding.device)
        encoding = torch.cat([tag_token, encoding], 1)
        
        #time-series length
        K = encoding.shape[1]

        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Output module
        output = torch.sigmoid(self.mlp(encoding))
    

        clipwise_output = output[:,0,:]
        
        framewise_output = output[:,1:,:]
        #clipwise_output = framewise_output.max(-2)[0]
        

        bp =framewise_output.size()[0]
        t_att = framewise_output
        t_att = t_att.view(int(bp/perturb_dim),
                perturb_dim, *t_att.shape[1:])

        return framewise_output, clipwise_output, t_att  
        # (last output dummy variable, placeholder for attention, to be implemented)



class PatchTransformer(nn.Module):

    def __init__(self, 
                n_in_channel: int,
                nclass: int, 
                d_model: int,
                q: int,
                v: int,
                h: int,
                N: int,
                d_ff: int = 512,
                attention_size: int = None,
                dropout: float = 0.3,
                chunk_mode: bool = True,
                pe: str = None,
                **kwargs):
        
        super(FullTransformer, self).__init__()
        
        self._d_model = d_model

        
        #self.final_freq = int(cfg.n_mels/reduce(lambda x,y: (x[0]*y[0],x[1]*y[1]),pool_size)[1])
        #self.cnn_out_dim = self.final_freq * nb_filters[-1]
        
        self._embedding = nn.Linear(cfg.n_mels, d_model)  
                
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      d_ff,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
                  
        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        #self.name = 'cnn_transformer'


        # self.mlp = nn.Sequential(
        #     nn.Linear(self._d_model, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, nclass)
        #     )
        
        self.mlp = nn.Linear(self._d_model, nclass)
               

    def load(self, filename=None, parameters=None):
        
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        


    def forward(self, x):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        in_shape = x.shape
        
        # the input tensor with the perturbation axis comes 
        # in the following dimensions:
        # (Batch x channel x Time x freq x perturb)
        if len(x.size())==5:
            x = x.permute(0,4,1,2,3)
            x = x.contiguous().view(-1,in_shape[1],in_shape[2],in_shape[3])
            perturb_dim = in_shape[-1]
        else:
            perturb_dim=1
        
               
        #reshape to from [Batch x channel x Time X freq] 
        #               --> [Batch x Time x Feature]   
        #                   (feature = channel x freq)
        b,ch,t,fr = x.shape
        x = x.transpose(1,2).reshape(b,t,-1)
        
        # Embedding module
        encoding = self._embedding(x)

        #concatenate the embedding with tag token (arbitrary constant vector) 
        tag_token = .2*torch.ones(encoding.shape[0],1,encoding.shape[2]).to(encoding.device)
        encoding = torch.cat([tag_token, encoding], 1)
        
        #time-series length
        K = encoding.shape[1]

        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Output module
        output = torch.sigmoid(self.mlp(encoding))
    

        clipwise_output = output[:,0,:]
        
        framewise_output = output[:,1:,:]
        #clipwise_output = framewise_output.max(-2)[0]
        
        bp =framewise_output.size()[0]
        t_att = framewise_output
        t_att = t_att.view(int(bp/perturb_dim),
                perturb_dim, *t_att.shape[1:])

        return framewise_output, clipwise_output, t_att  
        # (last output dummy variable, placeholder for attention, to be implemented)


