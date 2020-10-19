import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as pause
sigmoid_slope=1


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, pool_types=['avg', 'max'], num_layers=1):
        super(ChannelGate, self).__init__()
        #self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
        self.pool_types = pool_types

    def forward(self, x):

        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gate_c( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gate_c( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gate_c( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.gate_c( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw


        return channel_att_sum

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)

class TemporalGate(nn.Module):
    def __init__(self, gate_channel, nclass, n_freq = 8, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(TemporalGate, self).__init__()
        self.n_freq = n_freq
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',    nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
                        padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, nclass, kernel_size=(1,1)) )
        self.to_temporal_att_layer = nn.Conv2d( nclass, nclass, groups=nclass, kernel_size=(1,n_freq))
        #alternative lighter version of line above :
        #self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
        #self.gate_s.add_module( 'gate_s_freq_conv', nn.Conv2d(1, 10, kernel_size=(1,n_freq)) )


    def forward(self, in_tensor):
        
        sp_att  = self.gate_s( in_tensor )

        t_att = self.to_temporal_att_layer(F.relu_(sp_att)).squeeze(-1).permute(0,2,1)
        return sp_att, t_att


class BAM(nn.Module):
    def __init__(self, gate_channel, nclass,  reduction_ratio=16, n_freq=8):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel, reduction_ratio)
        #self.spatial_att = SpatialGate(gate_channel)
        self.temporal_att = TemporalGate(gate_channel, nclass, reduction_ratio, n_freq)
    
    def forward(self, x, perturb_dim):
        
        sp_att, t_att = self.temporal_att(x)
        
        # ---------------- overwrite the spatial attention by broacasting the t_att along freq axis ---
        #sp_att = sp_att.unsqueeze(-1).permute(0,-1,2,3,1)       
        sp_att = t_att.unsqueeze(-2).repeat(1,1,8,1).unsqueeze(1)
        # -------------------------------------------------------------------------

        batch_ptb,ch,tm,fr = x.shape 
        ch_att = self.channel_att(x).unsqueeze(2).unsqueeze(3).expand_as(x).unsqueeze(-1)
        x = x.unsqueeze(-1)
        
        # if perturb_dim > 0:
            
        #     x = x.view(int(batch_ptb/perturb_dim),perturb_dim,ch,tm,fr).unsqueeze(-1)
        #     x = x[:,0,:,:,:]

        #     #att = att.view(int(batch_ptb/perturb_dim),perturb_dim,10,tm,fr).unsqueeze(-1).permute(0,1,-1,3,4,2)
        #     att = att.view(int(batch_ptb/perturb_dim),perturb_dim,10,tm,fr).unsqueeze(-1).permute(0,1,-1,3,4,2)
            



        att =  torch.sigmoid(sigmoid_slope* (ch_att * sp_att) ) + 1
        

        return x * att, torch.sigmoid(sigmoid_slope*t_att)
