import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image

# Pretrained model used to transform, scale, and normalize images that will be passed
# into our style mode. This is a model designed for large scale image recognition. 
# Not sure why the basic tut includes this model yet
# model = models.vgg19(pretrained=True).features

# Assign the GPU to be used as a device (speed up training)
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Definine image loader
def image_loader(path):
    image = Image.open(path)
    
    # Defining image transform steps befor model feeding
    loader = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
    
    # Resize and convet image to tensor
    image = loader(image).unsqeeze(0)
    
    return image.to(device, torch.float)

# Define a class to provide more exacting feature extraction
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Targeted features - why?
        self.req_features = ['0', '5', '10', '19', '28']
        # Extracting first 29 features
        self.model = models.vgg19(pretrained=True).features[:29]
        
    def forward(self, x):
        # Feature storage array
        features = []
        
        # Iterate over model layers
        for layer_num, layer in enumerate(self.model):
            
            # Store activation in x
            x = layer(x)
            
            # Append activations and return the feature array
            if (str(layer_num) in self.req_features):
                features.append(x)
        
        return features


# Function to define loss calculation
def calc_content_loss(gen_feature, orig_feature):
    # Content loss calced by adding MSE loss between content and generated feature then added
    # to the content loss
    content_l = torch.mean((gen_feature - orig_feature) ** 2)
    return content_l

# Gram Matrix to calclate the style loss by doing the mean squared difference between the 
# generated and style Gram Matrices
def calc_style_loss(gen, style):
    # G-matrix parameters
    _, channel, height, width = gen.shape
    
    # Get both gram matrices
    G = torch.mm(gen.view(channel, height*width), gen.view(channel, height*width).t())
    A = torch.mm(style.view(channel, height*width), style.view(channel, height*width).t())
    
    # Calc mse loss between matrices
    style_l = torch.mean((G - A) ** 2)
    return style_l

# Now using the style & content loss we can calc the overall loss
def calc_loss(gen_features, orig_features, style_features) :
    style_loss, content_loss = 0
    for gen,cont,style in zip(gen_features, orig_features, style_features):
        content_loss = calc_content_loss(gen, cont)
        style_loss = calc_style_loss(gen, style)
        
    # Where we use alpha and beta weight coefficients
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss





