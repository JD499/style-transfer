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
# 3x3 conv on CxWxD matrix results in 9HWC^2 computations
# Network benefits: 

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
