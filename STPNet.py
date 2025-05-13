#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import torch
import numpy as np
import torch.nn as nn

import math
import random
# from tensorboardX import SummaryWriter
# writer = SummaryWriter('log_xbw')
from netCDF4 import Dataset 
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy.io import netcdf  
import scipy      
import scipy.signal
import matplotlib
#from mpl_toolkits.basemap import Basemap
from scipy.interpolate import Rbf    #RBF插值

import math
import pandas as pd
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import r2_score,mean_squared_error
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
# from pylab import * 


# In[2]:


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
#         print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
#         print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 4):
        super(ResBlock_CBAM,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )
        self.cbam = CBAM(channel=places*self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
#         print(x.shape)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# In[3]:


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual
        out = self.relu(out)

        return out

class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
#         print(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x
    
class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


# In[4]:


class DNN(nn.Module):
    def __init__(self,time_step=24):
        super(DNN, self).__init__()
        self.AvgPool2d = nn.AvgPool2d(kernel_size = 2)
        self.upsampling = nn.UpsamplingNearest2d(scale_factor = 2)
        self.fc = nn.Sequential(nn.Linear(2*128*2*2,4096),
                                nn.Tanh(),
                                nn.Linear(4096,4096),
                                nn.Tanh(),
                                nn.Linear(4096,4096),
#                                 nn.Tanh(),
#                                 nn.Linear(128,1)
                               )
        self.iod_fc = nn.Linear(4096,1)
        self.eiod_fc = nn.Linear(4096,1)
        self.wiod_fc = nn.Linear(4096,1)
#         self.month_fc = nn.Linear(128,12)
        
        self.time_step = time_step
        ####### stack4
        self.conv_stack4_1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=self.time_step*2,              # input height  gray just have one level
                out_channels=8*4*2,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.Tanh()
#             nn.ReLU()                              
        )
        self.conv_stack4_2 = nn.Sequential(         
            nn.Conv2d(8*4*2, 16*4*2, 3, 1, 1),
            nn.Tanh()
#             nn.ReLU()                
        )
        self.conv_stack4_3 = nn.Sequential(         
            nn.Conv2d(16*4*2, 32*4*2, 3, 1, 1),
            nn.Tanh()
#             nn.ReLU()               
        )
        self.conv_stack4_4 = nn.Sequential(         
            nn.Conv2d(32*4*2+self.time_step*4, 32, 3, 1, 1),
            nn.Tanh()
#             nn.ReLU()
        )
        #self.pad = nn.ZeroPad2d(padding=(28, 28, 4, 4))
        #1728 24*72
        num_channels=[3*9,3*9,3*9]
        self.time4_1_sst = TemporalConvNet(3*9, num_channels, kernel_size=2, dropout=0.2)
#         num_channels=[4096,4096,4096]
        self.time4_1_sss = TemporalConvNet(3*9, num_channels, kernel_size=2, dropout=0.2)
        self.batch_norm4_1 = nn.BatchNorm2d(32*4*2+time_step*4, affine=False)
        ### resCBAM1
#         self.res_attention1 = ResBlock_CBAM(in_places=(self.time_step)*2+32, places=20) 
        ####### stack3
        self.conv_stack3_1 = nn.Sequential(
            nn.Conv2d((self.time_step)*2+32,8*4*2,5,1,2),
            nn.Tanh()
#             nn.ReLU() 
        )
        self.conv_stack3_2 = nn.Sequential(
            nn.Conv2d(8*4*2,16*4*2,3,1,1),
            nn.Tanh()
#             nn.ReLU() 
        )
        self.conv_stack3_3 = nn.Sequential(
            nn.Conv2d(16*4*2,32*4*2,3,1,1),
            nn.Tanh()
#             nn.ReLU() 
        )
        self.conv_stack3_4 = nn.Sequential(
            nn.Conv2d(32*4*2+self.time_step*4+32,32,5,1,2),
            nn.Tanh()
#             nn.ReLU()
        )
        
        num_channels=[6*18,6*18,6*18]
        self.time3_1_sst = TemporalConvNet(6*18, num_channels, kernel_size=2, dropout=0.2)
#         num_channels=[4096,4096,4096]
        self.time3_1_sss = TemporalConvNet(6*18, num_channels, kernel_size=2, dropout=0.2)
        self.batch_norm3_1 = nn.BatchNorm2d(32*4*2+time_step*4+32, affine=False)
        
        
        ### resCBAM2
#         self.res_attention2 = ResBlock_CBAM(in_places=(self.time_step)*2+32, places=20) 
        ######### stack2
        self.conv_stack2_1 = nn.Sequential(
            nn.Conv2d((self.time_step)*2+32,8*4*2,5,1,2),
            nn.Tanh()
#             nn.ReLU() 
        )
        self.conv_stack2_2 = nn.Sequential(
            nn.Conv2d(8*4*2,16*4*2,3,1,1),
            nn.Tanh()
#             nn.ReLU() 
        )
        self.conv_stack2_3 = nn.Sequential(
            nn.Conv2d(16*4*2,32*4*2,3,1,1),
            nn.Tanh()
#             nn.ReLU() 
        )
        self.conv_stack2_4 = nn.Sequential(
            nn.Conv2d(32*4*2+self.time_step*4+32,32,5,1,2),
            nn.Tanh()
#             nn.ReLU()
        ) 
        
        num_channels=[12*36,12*36,12*36]
        self.time2_1_sst = TemporalConvNet(12*36, num_channels, kernel_size=2, dropout=0.2)
#         num_channels=[4096,4096,4096]
        self.time2_1_sss = TemporalConvNet(12*36, num_channels, kernel_size=2, dropout=0.2)
        self.batch_norm2_1 = nn.BatchNorm2d(32*4*2+time_step*4+32, affine=False)
        
        ### resCBAM2
#         self.res_attention3 = ResBlock_CBAM(in_places=(self.time_step)*2+32, places=20) 
        ####### stack1
        self.conv_stack1_1 = nn.Sequential(
            nn.Conv2d((self.time_step)*2+32,8*4*2,5,1,2),
            nn.Tanh()
#             nn.ReLU() 
        )
        self.conv_stack1_2 = nn.Sequential(
            nn.Conv2d(8*4*2,16*4*2,3,1,1),
            nn.Tanh()
#             nn.ReLU() 
        )
        self.conv_stack1_3 = nn.Sequential(
            nn.Conv2d(16*4*2,32*4*2,3,1,1),
            nn.Tanh()
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(2)
        )
        self.conv_stack1_4 = nn.Sequential(
            nn.Conv2d(32*4*2+self.time_step*4+32,128*2,5,1,2),
            nn.AdaptiveAvgPool2d(2),
            nn.Tanh()
        )
        
        num_channels=[24*72,24*72,24*72]
        self.time1_1_sst = TemporalConvNet(24*72, num_channels, kernel_size=2, dropout=0.2)
#         num_channels=[4096,4096,4096]
        self.time1_1_sss = TemporalConvNet(24*72, num_channels, kernel_size=2, dropout=0.2)
        self.batch_norm1_1 = nn.BatchNorm2d(32*4*2+time_step*4+32, affine=False)
#             nn.Tanh()
            
        # ) 

    def forward(self, x):
        x_stack1 = x
#         print(x_stack1.shape)
        x_stack2 = self.AvgPool2d(x_stack1)
#         print(x_stack2.shape)
        x_stack3 = self.AvgPool2d(x_stack2)
#         print(x_stack3.shape)
        x_stack4 = self.AvgPool2d(x_stack3)
#         print(x_stack4.shape)
        
        out = self.conv_stack4_1(x_stack4)
        out = self.conv_stack4_2(out)
        out = self.conv_stack4_3(out)
        
        x_stack4_sst = x_stack4[:,0::2,:]
        x_stack4_sss = x_stack4[:,1::2,:]
        x_shift4_sst = x_stack4_sst.permute([0, 3, 2, 1])
#         print(x_shift4_sst.shape)
        
        x_shift4_sst = x_shift4_sst.contiguous().view(x_stack4.shape[0],x_stack4.shape[2]*x_stack4.shape[3], int(x_stack4.shape[1]/2))
#         print()
        out_t_sst = self.time4_1_sst(x_shift4_sst)
        
        x_shift4_sss = x_stack4_sss.permute([0, 3, 2, 1])
        x_shift4_sss = x_shift4_sss.contiguous().view(x_stack4.shape[0],x_stack4.shape[2]*x_stack4.shape[3], int(x_stack4.shape[1]/2))
        out_t_sss = self.time4_1_sss(x_shift4_sss)        
        x_time_shift4_sst = out_t_sst.contiguous().view(x_stack4.shape[0],x_stack4.shape[3],x_stack4.shape[2], int(x_stack4.shape[1]/2))
        x_time_shift4_sss = out_t_sss.contiguous().view(x_stack4.shape[0],x_stack4.shape[3],x_stack4.shape[2], int(x_stack4.shape[1]/2))
        x_time_shift4_sst = x_time_shift4_sst.permute([0,3,2,1])
        x_time_shift4_sss = x_time_shift4_sss.permute([0,3,2,1])
        x_time_shift4_all = torch.empty((x_stack4.shape[0],x_stack4.shape[1],x_stack4.shape[2],x_stack4.shape[3])).to(device)
        x_time_shift4_all[:,0::2] = x_time_shift4_sst
        x_time_shift4_all[:,1::2] = x_time_shift4_sss
        
#         print(x_stack4.shape)
#         print(out.shape)
#         print(x_time_shift4_all.shape)
        out = torch.cat([out,x_time_shift4_all,x_stack4],dim=1)
#         print(out.shape)
        out = self.batch_norm4_1(out)
        out = self.conv_stack4_4(out)
        out_stack4_3 = self.upsampling(out)
        
        x_stack3_fh = torch.cat([x_stack3,out_stack4_3],dim=1)
#         x_stack3_fh = self.res_attention1(x_stack3_fh)
        out = self.conv_stack3_1(x_stack3_fh)
        out = self.conv_stack3_2(out)
        out = self.conv_stack3_3(out)
        
        x_stack3_sst = x_stack3[:,0::2,:]
        x_stack3_sss = x_stack3[:,1::2,:]
        x_shift3_sst = x_stack3_sst.permute([0, 3, 2, 1])
#         print(x_shift4_sst.shape)
        
        x_shift3_sst = x_shift3_sst.contiguous().view(x_stack3.shape[0],x_stack3.shape[2]*x_stack3.shape[3], int(x_stack3.shape[1]/2))
#         print()
        out_t_sst = self.time3_1_sst(x_shift3_sst)
        
        x_shift3_sss = x_stack3_sss.permute([0, 3, 2, 1])
        x_shift3_sss = x_shift3_sss.contiguous().view(x_stack3.shape[0],x_stack3.shape[2]*x_stack3.shape[3], int(x_stack3.shape[1]/2))
        out_t_sss = self.time3_1_sss(x_shift3_sss)        
        x_time_shift3_sst = out_t_sst.contiguous().view(x_stack3.shape[0],x_stack3.shape[3],x_stack3.shape[2], int(x_stack3.shape[1]/2))
        x_time_shift3_sss = out_t_sss.contiguous().view(x_stack3.shape[0],x_stack3.shape[3],x_stack3.shape[2], int(x_stack3.shape[1]/2))
        x_time_shift3_sst = x_time_shift3_sst.permute([0,3,2,1])
        x_time_shift3_sss = x_time_shift3_sss.permute([0,3,2,1])
        x_time_shift3_all = torch.empty((x_stack3.shape[0],x_stack3.shape[1],x_stack3.shape[2],x_stack3.shape[3])).to(device)
        x_time_shift3_all[:,0::2] = x_time_shift3_sst
        x_time_shift3_all[:,1::2] = x_time_shift3_sss
#         print(x_stack3.shape)
#         print(out.shape)
#         print(x_time_shift3_all.shape)
        out = torch.cat([out,x_time_shift3_all,x_stack3_fh],dim=1)
#         print(out.shape)
        
        
        # out += out_stack4_3
        # out += x_stack3
#         out = torch.cat([out,x_stack3_fh],dim=1)
        out = self.batch_norm3_1(out)
        out = self.conv_stack3_4(out)
        out_stack3_2 = self.upsampling(out)

        
        x_stack2_fh = torch.cat([x_stack2,out_stack3_2],dim=1)
#         x_stack2_fh = self.res_attention2(x_stack2_fh)
        out = self.conv_stack2_1(x_stack2_fh)
        out = self.conv_stack2_2(out)
        out = self.conv_stack2_3(out)
        # out += out_stack3_2
        # out += x_stack2
        x_stack2_sst = x_stack2[:,0::2,:]
        x_stack2_sss = x_stack2[:,1::2,:]
        x_shift2_sst = x_stack2_sst.permute([0, 3, 2, 1])
#         print(x_shift4_sst.shape)
        
        x_shift2_sst = x_shift2_sst.contiguous().view(x_stack2.shape[0],x_stack2.shape[2]*x_stack2.shape[3], int(x_stack2.shape[1]/2))
#         print()
        out_t_sst = self.time2_1_sst(x_shift2_sst)
        
        x_shift2_sss = x_stack2_sss.permute([0, 3, 2, 1])
        x_shift2_sss = x_shift2_sss.contiguous().view(x_stack2.shape[0],x_stack2.shape[2]*x_stack2.shape[3], int(x_stack2.shape[1]/2))
        out_t_sss = self.time2_1_sss(x_shift2_sss)        
        x_time_shift2_sst = out_t_sst.contiguous().view(x_stack2.shape[0],x_stack2.shape[3],x_stack2.shape[2], int(x_stack2.shape[1]/2))
        x_time_shift2_sss = out_t_sss.contiguous().view(x_stack2.shape[0],x_stack2.shape[3],x_stack2.shape[2], int(x_stack2.shape[1]/2))
        x_time_shift2_sst = x_time_shift2_sst.permute([0,3,2,1])
        x_time_shift2_sss = x_time_shift2_sss.permute([0,3,2,1])
        x_time_shift2_all = torch.empty((x_stack2.shape[0],x_stack2.shape[1],x_stack2.shape[2],x_stack2.shape[3])).to(device)
        x_time_shift2_all[:,0::2] = x_time_shift2_sst
        x_time_shift2_all[:,1::2] = x_time_shift2_sss
#         print(x_stack2.shape)
#         print(out.shape)
#         print(x_time_shift2_all.shape)
        out = torch.cat([out,x_time_shift2_all,x_stack2_fh],dim=1)
#         print(out.shape)
        out = self.batch_norm2_1(out)
        out = self.conv_stack2_4(out)
        out_stack2_1 = self.upsampling(out)

        
        x_stack1_fh = torch.cat([x_stack1,out_stack2_1],dim=1)
#         x_stack1_fh = self.res_attention3(x_stack1_fh)
        out = self.conv_stack1_1(x_stack1_fh)
        out = self.conv_stack1_2(out)
        out = self.conv_stack1_3(out)
#         print(out.shape)
#         print(x_stack1_fh.shape)
        x_stack1_sst = x_stack1[:,0::2,:]
        x_stack1_sss = x_stack1[:,1::2,:]
        x_shift1_sst = x_stack1_sst.permute([0, 3, 2, 1])
#         print(x_shift4_sst.shape)
        
        x_shift1_sst = x_shift1_sst.contiguous().view(x_stack1.shape[0],x_stack1.shape[2]*x_stack1.shape[3], int(x_stack1.shape[1]/2))
#         print()
        out_t_sst = self.time1_1_sst(x_shift1_sst)
        
        x_shift1_sss = x_stack1_sss.permute([0, 3, 2, 1])
        x_shift1_sss = x_shift1_sss.contiguous().view(x_stack1.shape[0],x_stack1.shape[2]*x_stack1.shape[3], int(x_stack1.shape[1]/2))
        out_t_sss = self.time1_1_sss(x_shift1_sss)        
        x_time_shift1_sst = out_t_sst.contiguous().view(x_stack1.shape[0],x_stack1.shape[3],x_stack1.shape[2], int(x_stack1.shape[1]/2))
        x_time_shift1_sss = out_t_sss.contiguous().view(x_stack1.shape[0],x_stack1.shape[3],x_stack1.shape[2], int(x_stack1.shape[1]/2))
        x_time_shift1_sst = x_time_shift1_sst.permute([0,3,2,1])
        x_time_shift1_sss = x_time_shift1_sss.permute([0,3,2,1])
        x_time_shift1_all = torch.empty((x_stack1.shape[0],x_stack1.shape[1],x_stack1.shape[2],x_stack1.shape[3])).to(device)
        x_time_shift1_all[:,0::2] = x_time_shift1_sst
        x_time_shift1_all[:,1::2] = x_time_shift1_sss
#         print(x_stack1.shape)
#         print(out.shape)
#         print(x_time_shift1_all.shape)
        out = torch.cat([out,x_time_shift1_all,x_stack1_fh],dim=1)
#         print(out.shape)
        out = self.batch_norm1_1(out)
        out = self.conv_stack1_4(out)
#         print(out.shape)
        # out += out_stack2_1
        # out += x_stack1
        #print('测试1:',out.shape)
        out_result = out.view(out.size(0), -1)
        #print('测试2:',out_result.shape)
        out_result1 = self.fc(out_result)
        out_result_iod = self.iod_fc(out_result1)
        out_result_eiod = self.eiod_fc(out_result1)
        out_result_wiod = self.wiod_fc(out_result1)
#         out_result_month = self.month_fc(out_result)
        
        return out_result_iod,out_result_eiod,out_result_wiod    # return x for visualization


# In[5]:


def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name)  
        else:  
            list_name.append(file_path)


# In[6]:


# 实现MyDatasets类
class MyDatasets(Dataset):
    def __init__(self, train_ssta_dir,label_dir,label_east_dir,label_west_dir):
        # 获取数据存放的dir
        # 例如d:/images/

        self.train_ssta_dir = train_ssta_dir
#         self.train_sssa_dir = train_sssa_dir
#         self.train_month_dir = train_month_dir
        self.label_dir = label_dir
        self.label_east_dir = label_east_dir
        self.label_west_dir = label_west_dir
        
        
        # 用于存放(image,label) tuple的list,存放的数据例如(d:/image/1.png,4)
        self.train_ssta_list = []
#         self.train_sssa_list = []
#         self.train_month_list = []
        self.label_list = []
        self.label_east_list = []
        self.label_west_list = []
        
        # 从dir--label的map文件中将所有的tuple对读取到image_target_list中
        # map.txt中全部存放的是d:/.../image_data/1/3.jpg 1 路径最好是绝对路径
        listdir(self.train_ssta_dir,self.train_ssta_list)
#         listdir(self.train_sssa_dir,self.train_sssa_list)
#         listdir(self.train_month_dir,self.train_month_list)
        listdir(self.label_dir,self.label_list)
        listdir(self.label_east_dir,self.label_east_list)
        listdir(self.label_west_dir,self.label_west_list)
        
        self.train_ssta_list.sort()
#         self.train_sssa_list.sort()
#         self.train_month_list.sort()
        self.label_list.sort()
        self.label_east_list.sort()
        self.label_west_list.sort()

    def __getitem__(self, index):
        train_ssta_pair = self.train_ssta_list[index]
#         train_sssa_pair = self.train_sssa_list[index]
#         train_month_pair = self.train_month_list[index]
        train_label_pair = self.label_list[index]
        train_label_east_pair = self.label_east_list[index]
        train_label_west_pair = self.label_west_list[index]
     
        # 按path读取图片数据，并转换为图片格式例如[3,32,32]
        # 可以用别的代替
#         img = cv2.imread(image_target_pair,0)
#         print(img.shape)
        train_ssta_data = np.load(train_ssta_pair)
#         train_sssa_data = np.load(train_sssa_pair)
#         train_month_data= np.load(train_month_pair)
        train_label_data = np.load(train_label_pair)
        train_label_east_data = np.load(train_label_east_pair)
        train_label_west_data = np.load(train_label_west_pair)
        
        return train_ssta_data,train_label_data,train_label_east_data,train_label_west_data
    def __len__(self):
        return len(self.label_list)


model = DNN(time_step=24)
model = model.to(device)
