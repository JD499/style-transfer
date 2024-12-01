# Import required packages
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

# In order to build matrix addition into sequential module need
# to create a module subclass for matrix addition
class MatrixAdder(nn.Module):
    def __init__(self, target):
        super(MatrixAdder, self, target).__init__()
        self.target = target
        
    def forward(self, inputs):
        A = inputs
        if A.size() != self.target.size():
            raise ValueError("Mismatched matrix sizes")
        
        return A + self.target            

# stride 2 9x9 conv block input 3x256x256 matrix
# Output shape:
# stride 1/2 3x3 block
# stide 1/2 9x9 block

# Conv block with variable stride & kernel

# strid 2 3x3 conv block

# Downsampling Block - allows us to use larger network for same computational cost
# C - number of filters (how many to use?)
# 3x3 conv on CxWxH matrix results in 9HWC^2 computations
# 3x3 conv with D downsample gives CDxW/DxH/D gives also 9HWC^2 comps
# Network benefits: We want one pixel to have a larger receptive field (i.e. range of
# data that the pixel affects) which can be done by stacking layers (computationally expensive)
# downsampling by some factor D increases the receptive field by 2D
# This block will take the input and per the paper the first conv is 9x9
class DownsampleBlock(nn.Module):
    def __init__(self):
        super(DownsampleBlock, self).__init__()
        self.downblock = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=3,
                      kernel_size=(9,9),
                      stride=2),
            nn.Conv2d(in_channels=3,
                      out_channels=3,
                      kernel_size=(3,3),
                      stride=2)
        )
        
    def forward(self, x):
        output = self.downblock(x)
        return output

# Upsampling Block
# The paper proposes an upsampling process to create high resolution
# results. The paper has options for a dynamic upsampling factor f. Instead
# Instead of the factor we will be following the design the paper proposes
# to use two stride = 1/2 convolution layers to upsample the image
class UpsampleBlock(nn.Module):
    def __init__(self):
        super(UpsampleBlock, self).__init__()
        self.upblock = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=3,
                      kernel_size=(3,3),
                      stride= 0.5),
            # This is the last layer and per the paper will have a 9x9 kernel
            nn.Conv2d(in_channels=3,
                      out_channels=3,
                      kernel_size=(9,9),
                      stride= 0.5)
        )
    
    def forward(self, x):
        output = self.upblock(x)
        return output

# Residual block
# temp = stored input imag
# img -> 3x3 conv -> Batch N -> ReLU -> 3x3 conv -> Batch N -> add image -> ReLu -> output
# Current confusion about the in/out channel sizes (i think should be 3 throughout)
class ResidualBlock(nn.Module):
    # An image input
    def __init__(self, target):
        super(ResidualBlock, self, target).__init__()
        self.temp = target
        self.resblock = nn.Sequential(
            # 3 channel and 1 filter per channel 3x3 conv2d step
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3)),
            # 3 channels and 1 filter = 3 features?
            nn.BatchNorm1d(num_features=3),
            # ReLU activation specifed by source
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3)),
            nn.BatchNorm1d(num_features=3),
            # Should take BatchNorm1d results and add to target matrix
            MatrixAdder(self.temp),
            nn.ReLU()
        )
        
    def forward(self, x):
        output = self.resblock(x)
        
        return output
