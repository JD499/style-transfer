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
            nn.ReflectionPad2d(2),
            # Paper specifies that fraction tranposes are used 
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=(9,9),
                      stride=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU()
        )
        
    def forward(self, x):
        print(x.size())
        print('moved through downsample')
        output = self.downblock(x)
        print(output.size())
        return output

# Upsampling Block
# The paper proposes an upsampling process to create high resolution
# results. The paper has options for a dynamic upsampling factor f. Instead
# Instead of the factor we will be following the design the paper proposes
# to use two stride = 1/2 convolution layers to upsample the image
# TO do this we use a fixed upsample factor of 2
class UpsampleBlock(nn.Module):
    def __init__(self):
        super(UpsampleBlock, self).__init__()
        self.upblock = nn.Sequential(
            # nn.Upsample(mode='nearest', scale_factor=2),            
            # nn.ReflectionPad2d(2),
            nn.ConvTranspose2d(in_channels=64,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=2,
                      padding=1,
                      output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,
                      out_channels=3,
                      kernel_size=(9,9),
                      stride= 2,
                      padding=4,
                      output_padding=1),
            # To restrict rbg data back to [0,255] range
            ScaledTanh()
        )
    
    def forward(self, x):
        print('moved through upsample')
        output = self.upblock(x)
        print(output.size())
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
            nn.ReflectionPad2d(2),
            # 64 channel and 1 filter per channel 3x3 conv2d step
            nn.Conv2d(channels, channels, kernel_size=(3,3), stride=1),
            # Instance Norms suggested by https://arxiv.org/pdf/1708.04538
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            # nn.ReflectionPad2d(2),
            # ReLU activation specifed by source
            nn.Conv2d(channels, channels, kernel_size=(3,3), stride=1),
            nn.InstanceNorm2d(channels, affine=True),
        )
        
    def forward(self, x):
        residual = x
        print(residual.size())
        print('moved through residual')
        first_out = self.resblock(x)
        print(first_out.size())
        skip = first_out + residual
        return torch.relu(skip)
       
class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        self.model = nn.Sequential(
            DownsampleBlock(),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            UpsampleBlock()
        )
    
    def forward(self, x):
        out = self.model(x)
        return out

# Gram Matrix Calc Utility
def gram_matrix(x):
    # reshape x into a CxWH matrix
    _, c, w, h = x.size()
    R = x.view((c, w*h))
    print(R.size()) 

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
    def __init__(self, alpha, beta, gamma):
        super(PerceptualLoss, self).__init__()
        self.style_loss = StyleLoss()
        self.feature_loss = FeatureLoss()
        self.tvr = TVR()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        # style, content, generated packed into x
        style,content,gen = x
        perceptual_loss = self.alpha * self.style_loss((style,gen)) + self.beta * self.feature_loss((content, gen)) + self.gamma * self.tvr(gen)
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
    def __init__(self):
        super(StyleLoss, self).__init__()
        # Only intersted in a certain set of features 3,8,15,24
        self.feature_set = ['3', '8', '15', '24']
        self.model = models.vgg.vgg19(pretrained=True).features[:29]
        
    def forward(self, x):
        # Style, gen packed into x
        style, gen = x
        style_loss = 0 
        print('calc style loss')
        # Extract desired features
        for name, module in enumerate(self.model):
            gen = module(gen)
            
            # The style image also needs to pass through the vgg activation layers
            S = gram_matrix(module(style))
            
            # Calculate the gram matrix at the desired layers
            if str(name) in self.feature_set:
                G = gram_matrix(gen)
                style_loss += torch.square(torch.linalg.norm(G - S))
                
        return style_loss
        

# Content Loss
# Calculating the feature reconstruction loss in order to capture
# aspect like texture and shape that are otherwise lost
# Here we take the euclidean norm of the activation output vectors between
# the generated image and the target content image
class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.model = models.vgg.vgg19(pretrained=True).features[:29]
        self.feature_set = ['8']

    def forward(self, x):
        # content, genereated is passed into the FeatureLoss
        content, gen = x 
        content_loss = 0
        
        print('calc feature loss')
        for name, module in enumerate(self.model):
            if str(name) in self.feature_set:
                gen = module(gen)
                # Extract shape to scale
                c,w,h = gen.shape()
                scale_factor = c * w * h                
                
                content_activation = module(content)
                # Calculate the MSE square error of the scaled vector norm
                content_loss += torch.div(
                    torch.square(torch.linalg.vector_norm(gen - content_activation)),
                    scale_factor
                )
                
        return content_loss

# Helper from src3 to convert img back from tensor
def recover_image(img):
    return (
        (
            img *
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) *
        255.
    ).clip(0, 255).astype(np.uint8)

# Training the model
def main():
    from torchvision import datasets
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    # Training on COCO17 dataset
    IMAGE_SIZE = 256
    # TEMP : Set to 1 at the moment because we are testing that the network will even train
    BATCH_SIZE = 1
    # STYLE
    STYLE_IMAGE = './style.jpg'
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup image transformation for dataloader
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
        # mean and std taken from ImageNet norm & std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
    
    # Gets the coco dataset URLs. Each url needs to be grabbed and read still 
    train_data = datasets.ImageFolder('./train', transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    
    # Setup PerceptuaLoss network
    with torch.no_grad():
        loss_network = PerceptualLoss(1.0, 1.0, 1.0)
        loss_network.to(device)
    loss_network.eval()
        
    # Setup style image
    style_img = Image.open(STYLE_IMAGE).convert('RGB')
    with torch.no_grad():
        style_img_tensor = transforms.Compose([
            transforms.Resize(IMAGE_SIZE*2),
            transforms.ToTensor(),
            transforms.Normalize(
            # mean and std taken from ImageNet norm & std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])(style_img).unsqueeze(0)
        style_img_tensor = style_img_tensor.to(device)
        print(style_img_tensor.size())
        exit()
    # Check that image was properly converted to tensor
    # plt.imshow(recover_image(style_img_tensor.cpu().numpy())[0])

    # Model setup
    model = StyleTransferModel()
    model.to(device)
    torch.set_default_tensor_type('torch.FloatTensor')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # Train the model
    model.train()
    count = 0
    total_loss = 0
    
    while True:
        for x, _ in train_loader:
            count += 1
            optimizer.zero_grad() 
            
            # Pass image through transformer 
            x = x.to(device)
            y = model(x)
            
            # Pass through the loss function and pass in style
            # content and generated
            total_loss += loss_network((style_img_tensor, x, y))
            print(total_loss)

            if count == 1:
                return
    
    return
                
            
if __name__ == "__main__":
    main()