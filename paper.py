# Import required packages
from collections import namedtuple
import time
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

# Struggle #1
# Paper 2 does not provide any code for the architechture that it proposes. Repeated mention of supplmentary materials 
# are made but supplmenetary materials were not available with the arxiv preprint paper. The use of fraction convolutions 
# is mentioned with a stride of 1/2. The only fractional convolutions available are convTranspose2d where the fraction 
# 1/2 presented by the paper must be flipped. I also had to add padding within the residual blocks which is not 
# specifically stated in the original presentation of the residual block archtechture. This is because the conv2d blocks
# will naturally reduce the size of the matrix. Though the conv takes padding as an argument I used reflection padding instead
# of standard zero padding because the points on the edge should be entirely influenced by the styles and features of 
# the surrounding area. 

# Struggl #2
# First model implementation used three modules where the style and feature loss were seperate and acted as seperate modules
# joined in another module that computed the perceptual loss. The way that this was implemented required the retain_graph option
# to be set to True as well as repeated tensor cloning to avoid inplace operations. When doing this the entire computation graph 
# along with the cloned tensors (larges ones around 64,60000 for the gram matrices) started to eat GPU memory and within 20 iterations 
# broke 20gb of allocated memory which forced the training to stop on a 4060 limited to 16gb of available memory. 

# To solve this I will be taking some inspiration from src 3 which computes only the relu activation values and returns those
# and then computes the actual perceptual loss within the training loop.

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
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=2),
            nn.InstanceNorm2d(64, affine=True),
            # nn.BatchNorm2d(64),
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
    def __init__(self):
        super(UpsampleBlock, self).__init__()
        self.upblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=2,
                      padding=1,
                      output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,
                      out_channels=3,
                      kernel_size=(9,9),
                      stride= 2,
                      padding=4,
                      output_padding=1),
            nn.InstanceNorm2d(3, affine=True), 
            # nn.BatchNorm2d(3),            
            # To restrict rbg data back to [0,255] range
            # ScaledTanh()
            nn.Tanh()
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
        self.last_relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        first_out = self.resblock(x)
        skip = first_out + residual
        return self.last_relu(skip)
       
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
    b, c, w, h = x.size()
    r = x.view(b, c, w*h)
    r_t = r.transpose(1,2)
    # Calc gram matrix using reshaped tensor
    gram = torch.bmm(r,r_t) / (c*w*h)
    return gram

# Perceptual Loss
# We are grabbing some layers from the VGG16 pretrained model
# and evaluating our style transfer based on the perceptual loss optimization
# Here we take our StyleLoss & Content Loss outputs, sum them and add the total variance
# regularizer. Here target is the image to be styled and the style image is the style reference
# alpha, beta, gamma refer to the loss weights
# class PerceptualLoss(nn.Module):
#     def __init__(self, alpha, beta, gamma):
#         super(PerceptualLoss, self).__init__()
#         self.style_loss = StyleLoss()
#         self.feature_loss = FeatureLoss()
#         self.tvr = TVR()
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma

#     def forward(self, x):
#         # style, content, generated packed into x
#         style,content,gen = x
#         style_loss = self.style_loss((style.clone(), gen.clone()))
#         print(f'Style Loss: {style_loss}')
#         feature_loss = self.feature_loss((content.clone(), gen.clone()))
#         print(f'Feature Loss: {feature_loss}')
#         tvr_loss = self.tvr(gen.clone())
#         print(f'TVR Loss: {tvr_loss}')
        
#         perceptual_loss = self.alpha * style_loss + self.beta * feature_loss + self.gamma * tvr_loss
#         print(f'Preceptual Loss: {perceptual_loss}')
#         return perceptual_loss
        
# Total Variance Regularizer
# class TVR(nn.Module):
#     def __init__(self):
#         super(TVR, self).__init__()
    
def tvr(x):    
    return torch.sum(torch.abs(x[:,:,:,:-1]-x[:,:,:,1:])) + torch.sum(torch.abs(x[:,:,:-1,:]-x[:,:,1:,:]))

# Style Loss
# MSE of gram matrix differences between the generated and style images
# Assume that a CxHxW matrix is being supplied to Style Loss
# Requires style matrix to be provided in order to generate the loss value which is the 
# mean square difference of the gram matrix of the generated and the style content
# class StyleLoss(nn.Module):

# Next Implementation
# Reduce number of modules and inplace operations by returning only the activation outputs
# and handling loss calculations within the training loop
LossOutput = namedtuple(
    "LossOutput", ['relu1', 'relu2', 'relu3', 'relu4']
)
class LossNetwork(nn.Module):
    def __init__(self):
        # super(StyleLoss, self).__init__()
        super(LossNetwork, self).__init__()
        # Only intersted in a certain set of features 3,8,15,24
        # self.feature_set = ['3', '8', '15', '24']
        self.layer_mapping = {
            '3' : 'relu1',
            '8' : 'relu2',
            '15': 'relu3',
            '24': 'relu4'
        }
        self.model = models.vgg.vgg19(pretrained=True).features[:29]
        
    def forward(self, x):
        # Style, gen packed into x
        output = {}
        # style_loss = 0 
        # Extract desired features
        for name, module in self.model._modules.items():
            x = module(x)
            # style = module(style.clone())
            if name in self.layer_mapping.keys():
                output[self.layer_mapping[name]] = x
            # Calculate the gram matrix at the desired layers
            # if str(name) in self.feature_set:
            #     G = gram_matrix(gen)
            #     S = gram_matrix(style)
            #     style_loss += torch.square(torch.linalg.norm(G - S))
                
        return LossOutput(**output)
        

# Content Loss
# Calculating the feature reconstruction loss in order to capture
# aspect like texture and shape that are otherwise lost
# Here we take the euclidean norm of the activation output vectors between
# the generated image and the target content image
# class FeatureLoss(nn.Module):
#     def __init__(self):
#         super(FeatureLoss, self).__init__()
#         self.model = models.vgg.vgg19(pretrained=True).features[:29]
#         self.feature_set = ['8']

#     def forward(self, x):
#         # content, genereated is passed into the FeatureLoss
#         content, gen = x 
#         content_loss = 0
        
#         for name, module in enumerate(self.model):
#             if str(name) in self.feature_set:
#                 gen = module(gen)
#                 content = module(content)
                
#                 print(gen.size())
#                 print(content.size())
#                 # Extract shape to scale
#                 _,c,w,h = gen.size()
#                 scale_factor = c * w * h                
                
#                 # Calculate the MSE square error of the scaled vector norm
#                 content_loss += torch.div(
#                     torch.square(torch.linalg.vector_norm(gen - content)),
#                     scale_factor
#                 )
                
#         return content_loss

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
    
def save_debug_image(tensor_orig, tensor_transformed, filename):
    assert tensor_orig.size() == tensor_transformed.size()
    result = Image.fromarray(recover_image(tensor_transformed.cpu().numpy())[0])
    orig = Image.fromarray(recover_image(tensor_orig.cpu().numpy())[0])
    new_im = Image.new('RGB', (result.size[0] * 2 + 5, result.size[1]))
    new_im.paste(orig, (0,0))
    new_im.paste(result, (result.size[0] + 5,0))
    new_im.save(filename)

# Training the model
def main():
    from torchvision import datasets
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    # Training on COCO17 dataset
    IMAGE_SIZE = 256
    # TEMP : Set to 1 at the moment because we are testing that the network will even train
    BATCH_SIZE = 4
    # STYLE
    STYLE_IMAGE = './style.jpg'
    # WEIGHTS
    FEATURE_WEIGHT = 0.5
    STYLE_WEIGHT = 1e5
    TVR_WEIGHT = 1e-5
    # Learning Rate
    LR = 0.001
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup image transformation for dataloader
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
        # mean and std taken from ImageNet norm & std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
    
    # Gets the coco dataset URLs. Each url needs to be grabbed and read still 
    train_data = datasets.ImageFolder('./train', transform)
    print('loading data')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    print('data loaded')
    # Setup PerceptuaLoss network
    with torch.no_grad():
        loss_network = LossNetwork()
        loss_network.to(device)
    loss_network.eval()
        
    # Setup style image
    style_img = Image.open(STYLE_IMAGE).convert('RGB')
    with torch.no_grad():
        style_img_tensor = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
            # mean and std taken from ImageNet norm & std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])(style_img).unsqueeze(0)
        style_img_tensor = style_img_tensor.to(device)
    print(style_img_tensor.size())
    # Check that image was properly converted to tensor
    # plt.imshow(recover_image(style_img_tensor.cpu().numpy())[0])

    # Model setup
    model = StyleTransferModel()
    model.to(device)
    torch.set_default_tensor_type('torch.FloatTensor')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)

    # Train the model
    model.train()
    count = 0
    max_epoch = 4
    epoch = 1
    mse = nn.MSELoss()

    # DEBUG
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.set_per_process_memory_fraction(0.85)
    
    # Calc style gram matrix
    with torch.no_grad():
        style_feature_loss = loss_network(style_img_tensor)
        style_gram = [gram_matrix(y) for y in style_feature_loss]
    
    # Initialize Loss Tracking
    total_feature_loss = 0
    total_style_loss = 0
    total_tvr_loss = 0
    
    while True:
        for x, _ in train_loader:
            count += 1
            optimizer.zero_grad() 
            
            # Pass image through transformer 
            x = x.to(device)
            y = model(x)
            
            # Isolate content
            with torch.no_grad():
                x_feature = x.detach()
                
            # Calc feature losses
            y_loss = loss_network(y)
            x_loss = loss_network(x_feature)
            
            with torch.no_grad():
                x_feature_loss = x_loss[1].detach()
            
            feature_loss = FEATURE_WEIGHT * mse(y_loss[1], x_feature_loss)
            
            # Calc style losses
            style_loss = 0
            # Iterate over the four layers where we want to calculate Gram Matrix MSE
            for i in range(0,3):
                y_gram = gram_matrix(y_loss[i])
                style_loss += mse(y_gram, style_gram[i].expand_as(y_gram))
            style_loss = STYLE_WEIGHT * style_loss
            
            # Calc TVR loss
            tvr_loss = TVR_WEIGHT * tvr(y)
            
            # Backpropogate
            total_loss = feature_loss + style_loss + tvr_loss
            total_loss.backward()
            optimizer.step()
            
            # Update total counts
            total_feature_loss += feature_loss
            total_style_loss += style_loss
            total_tvr_loss += tvr_loss
            
            # Debug msg
            if count % 1000 == 0:
                msg = f'{time.ctime()} [{count} / {max_epoch * 4000}] feature:{total_feature_loss} style:{total_style_loss} tvr:{total_tvr_loss} total:{total_feature_loss + total_style_loss + total_tvr_loss}'
                model.eval()
                y = model(x)
                save_debug_image(x, y.detach(), "./debug/{}.png".format(count))
                model.train()
                print(msg)
                
            if count % 10250 == 0:
                optimizer = torch.optim.Adam(model.parameters(), LR * 0.1)
                epoch += 1
                
            if epoch >= max_epoch:
                torch.save(model, './models/model.pt')
                return

                
            
if __name__ == "__main__":
    main()