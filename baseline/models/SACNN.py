import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.cbam import CBAM

from pdb import set_trace as pause
from time import time

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,dil=(1,1),
                kernel_dim=(3,3),stride=(1,1),pad=(1,1)):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_dim, stride=stride,
                              padding=pad, dilation=dil, bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_dim, stride=stride,
                              padding=pad, dilation=dil, bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x    



class CNN(nn.Module):
    def __init__(self, nclass, stride, pad, dilation,
                pool_size, pool_type='max',n_in_channel=1,  nb_filters=[64,128,256,512],  **kwargs):
        
        super(CNN, self).__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.conv_block1 = ConvBlock(in_channels=n_in_channel, out_channels=nb_filters[0],
                                     pad=pad[0],stride=stride[0],dil=dilation[0])
        self.conv_block2 = ConvBlock(in_channels=nb_filters[0], out_channels=nb_filters[1],
                                     pad=pad[1],stride=stride[1],dil=dilation[1])
        
        self.conv_block3 = ConvBlock(in_channels=nb_filters[1], out_channels=nb_filters[2],
                                     pad=pad[2],stride=stride[2],dil=dilation[2])

        self.conv_block4 = ConvBlock(in_channels=nb_filters[2], out_channels=nb_filters[3],
                                     pad=pad[3],stride=stride[3],dil=dilation[3])

        self.fc = nn.Linear(nb_filters[-1], nclass, bias=True)

        self.init_weights()
        
        #self.strong_target_training = strong_target_training

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, x):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        
        #x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=self.pool_size[0], pool_type=self.pool_type)
        x = self.conv_block2(x, pool_size=self.pool_size[1], pool_type=self.pool_type)
        x = self.conv_block3(x, pool_size = self.pool_size[2], pool_type=self.pool_type)
        tf_maps = self.conv_block4(x, pool_size= self.pool_size[3], pool_type=self.pool_type)
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''
        
        (framewise_vector, _) = torch.max(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        #framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        # Clipwise prediction
        #if self.strong_target_training:
        #    # Obtained by taking the maximum framewise predictions
        #    clipwise_output, _ = torch.max(framewise_output, dim=1)
        #    
        #else:
        #    # Obtained by applying fc layer on aggregated framewise_vector
        #    (aggregation, _) = torch.max(framewise_vector, dim=2)
        #    clipwise_output = torch.sigmoid(self.fc(aggregation))
        (aggregation, _) = torch.max(framewise_vector, dim=2)
        clipwise_output = torch.sigmoid(self.fc(aggregation))    
        return framewise_output, clipwise_output


class SACNN(nn.Module):
    def __init__(self, nclass, stride, pad, dilation,
                pool_size, pool_type='max',n_in_channel=1,  nb_filters=[64,128,256,512], **kwargs):
        
        super(SACNN, self).__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type
        
        self.conv_block1 = ConvBlock(in_channels=n_in_channel, out_channels=nb_filters[0],
                                     pad=pad[0],stride=stride[0],dil=dilation[0])
        self.conv_block2 = ConvBlock(in_channels=nb_filters[0], out_channels=nb_filters[1],
                                     pad=pad[1],stride=stride[1],dil=dilation[1])
        
        self.conv_block3 = ConvBlock(in_channels=nb_filters[1], out_channels=nb_filters[2],
                                     pad=pad[2],stride=stride[2],dil=dilation[2])

        self.conv_block4 = ConvBlock(in_channels=nb_filters[2], out_channels=nb_filters[3],
                                     pad=pad[3],stride=stride[3],dil=dilation[3])
        self.cbam = CBAM(nb_filters[3],16)
        self.fc = nn.Linear(nb_filters[-1], 1, bias=True)

        self.init_weights()
        
        #self.strong_target_training = strong_target_training

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, x):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        #x = torch.randn(1,1,864,64).cuda()
        
        interpolate_ratio = 8
        in_shape = x.shape
        if len(x.size())==5:
            x = x.permute(0,4,1,2,3)
            x = x.contiguous().view(-1,in_shape[1],in_shape[2],in_shape[3])
            perturb_dim = in_shape[-1]
        else:
            perturb_dim=1
            #no perturbation in data        
        #x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=self.pool_size[0], pool_type=self.pool_type)
        x = self.conv_block2(x, pool_size=self.pool_size[1], pool_type=self.pool_type)
        x = self.conv_block3(x, pool_size = self.pool_size[2], pool_type=self.pool_type)
        tf_maps = self.conv_block4(x, pool_size= self.pool_size[3], pool_type=self.pool_type)
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''
        

        tf_maps, tf_att = self.cbam(tf_maps, perturb_dim)
        #tf_maps : (batch_size, feature_maps, time_steps, frq_bins, nclass)
            
        #tf_att : (batch_size, augment_dim, feature_maps, time_steps, frq_bins, nclass))
        
        t_att = tf_att.max(axis=4)

        (framewise_vector, _) = torch.max(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        framewise_output = torch.sigmoid(self.fc(framewise_vector.permute([0,3,2,1])))
        #framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, classes_num, frames_num, 1)'''

        framewise_output = framewise_output.squeeze(-1).permute([0,2,1])
        #(batch_size, frames_num, classes_num)  

        # Clipwise prediction
        #if self.strong_target_training:
        # Obtained by taking the maximum framewise predictions
        clipwise_output, _ = torch.max(framewise_output, dim=1)

        #    
        #else:
        #    # Obtained by applying fc layer on aggregated framewise_vector
        #    (aggregation, _) = torch.max(framewise_vector, dim=2)
        #    clipwise_output = torch.sigmoid(self.fc(aggregation))
        #(aggregation, _) = torch.max(framewise_vector, dim=2)
        #clipwise_output = torch.sigmoid(self.fc(aggregation))    
        return framewise_output, clipwise_output, t_att[0]



