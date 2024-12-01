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

# Src 1 = https://arxiv.org/pdf/1708.04538
#   - res block with instance norm
# Src 2 = https://arxiv.org/pdf/1603.08155
#   - res block orig [44]
# Src 3 = https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
#   - guidance on how to implement the model
        

# Scaling tanh to output in [0,255] range
# f(x) = |255tanh(x)|
class ScaledTanh(nn.Module):
    def __init__(self):
        super(ScaledTanh, self).__init__()
    
    def forward(self, x):
        scaled = torch.abs(255 * torch.tanh(x))
        return scaled
    
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
                      out_channels=32,
                      kernel_size=(9,9),
                      stride=2),
            nn.BatchNorm1d(num_features=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=2),
            nn.ReLU()
        )
        
    def forward(self, x):
        output = self.downblock(x)
        return output

# Upsampling Block
# The paper proposes an upsampling process to create high resolution
# results. The paper has options for a dynamic upsampling factor f. Instead
# Instead of the factor we will be following the design the paper proposes
# to use two stride = 1/2 convolution layers to upsample the image
# TO do this we use a fixed upsample factor of 2
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample = 2):
        super(UpsampleBlock, self).__init__()
        self.upsample = upsample
        self.upblock = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride= 0.5),
            nn.BatchNorm1d(num_features=3),
            nn.ReLU(),
            # This is the last layer and per the paper will have a 9x9 kernel
            nn.Conv2d(in_channels=32,
                      out_channels=3,
                      kernel_size=(9,9),
                      stride= 0.5),
            # To restrict rbg data back to [0,255] range
            ScaledTanh()
        )
    
    def forward(self, x):
        output = self.upblock(x)
        return output

# Residual block
# temp = stored input imag
# img -> 3x3 conv -> Batch N -> ReLU -> 3x3 conv -> Batch N -> add image -> ReLu -> output
# Current confusion about the in/out channel sizes (i think should be 3 throughout)
# Essentially we are adding a skip block
class ResidualBlock(nn.Module):
    # An image input
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.resblock = nn.Sequential(
            # 3 channel and 1 filter per channel 3x3 conv2d step
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            # Instance Norms suggested by https://arxiv.org/pdf/1708.04538
            nn.InstanceNorm2d(channels, affine=True),
            # ReLU activation specifed by source
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            # Should take BatchNorm1d results and add to target matrix
            nn.ReLU()
        )
        
    def forward(self, x):
        residual = x
        output = self.resblock(x)
        return output + residual
       
class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        self.model = nn.Sequential(
            DownsampleBlock(),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            UpsampleBlock()
        )

# Gram Matrix Calc Utility
def gram_matrix(x):
    # reshape x into a CxWH matrix
    c, w, h = x.shape()
    R = torch.reshape(c, '-1') 

    # Calc gram matrix using reshaped tensor
    G = torch.div(torch.matmul(R, R.t()), c*w*h)
    return G

# Perceptual Loss
# We are grabbing some layers from the VGG16 pretrained model
# and evaluating our style transfer based on the perceptual loss optimization
# Here we take our StyleLoss & Content Loss outputs, sum them and add the total variance
# regularizer. Here target is the image to be styled and the style image is the style reference
# alpha, beta, gamma refer to the loss weights
class PerceptualLoss(nn.Module):
    def __init__(self, target, style, alpha, beta, gamma):
        super(PerceptualLoss, self).__init__()
        self.style_loss = StyleLoss(style)
        self.feature_loss = FeatureLoss(target)
        self.tvr = TVR()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        perceptual_loss = self.alpha * self.style_loss(x) + self.beta * self.feature_loss(x) + self.gamma * self.tvr(x)
        return perceptual_loss
        
# Total Variance Regularizer
class TVR(nn.Module):
    def __init__(self):
        super(TVR, self).__init__()
    
    def forward(self, x):
        c,h,w = x.shape()
        tv_h = torch.pow(x[:,1:,:]-x[:,:-1,:], 2).sum()
        tv_w = torch.pow(x[:,:,1:]-x[:,:,:-1], 2).sum()
        return (tv_h+tv_w)/(c*h*w)

# Style Loss
# MSE of gram matrix differences between the generated and style images
# Assume that a CxHxW matrix is being supplied to Style Loss
# Requires style matrix to be provided in order to generate the loss value which is the 
# mean square difference of the gram matrix of the generated and the style content
class StyleLoss(nn.Module):
    def __init__(self, style):
        super(StyleLoss, self).__init__()
        # Only intersted in a certain set of features 3,8,15,24
        self.feature_set = ['3', '8', '15', '24']
        self.style = style
        self.model = models.vgg.vgg19(pretrained=True).features[:29]
        
    def forward(self, x):
        style_loss = 0 
        # Extract desired features
        for name, module in enumerate(self.model):
            x = module(x)
            
            # The style image also needs to pass through the vgg activation layers
            S = gram_matrix(module(self.style))
            
            # Calculate the gram matrix at the desired layers
            if str(name) in self.feature_set:
                G = gram_matrix(x)
                style_loss += torch.square(torch.linalg.norm(G - S))
                
        return style_loss
        

# Content Loss
# Calculating the feature reconstruction loss in order to capture
# aspect like texture and shape that are otherwise lost
# Here we take the euclidean norm of the activation output vectors between
# the generated image and the target content image
class FeatureLoss(nn.Module):
    def __init__(self, target):
        super(FeatureLoss, self).__init__()
        self.target = target
        self.model = models.vgg.vgg19(pretrained=True).features[:29]
        self.feature_set = ['8']

    def forward(self, x):
        content_loss = 0
        
        for name, module in enumerate(self.model):
            if str(name) in self.feature_set:
                x = module(x)
                # Extract shape to scale
                c,w,h = x.shape()
                scale_factor = c * w * h                
                
                content_activation = module(self.content)
                # Calculate the MSE square error of the scaled vector norm
                content_loss += torch.div(
                    torch.square(torch.linalg.vector_norm(x - content_activation)),
                    scale_factor
                )
                
        return content_loss

