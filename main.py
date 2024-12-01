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

# LOG: Struggles
"""
1. Upon getting the network working and figuring out the renormalization the network would
run and nothing would happen to theinput image. It turned out that by setting both the weight
"""

# Assign user's device for optimal runtime (GPU or CPU)
def set_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Use MPS (Metal Performance Shaders) if available
    elif torch.cuda.is_available():
        return torch.device("cuda")  # Use CUDA (GPU) if available
    else:
        return torch.device("cpu")  # Otherwise, use CPU

# Define image transformations (USED IN LOAD_IMAGE FUNCTION)
transform = transforms.Compose([
    # Convert the image to a tensor : t
    transforms.ToTensor(),  
    # mean (from ImageNet dataset) = (0.485, 0.456, 0.406)
    # sd   (from ImageNet dataset) = (0.229, 0.224, 0.225)
    # t_n = (t - mean) / std
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize the image
])

# Load & preprocess an image
def load_image(img_path, transform=None, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')  # Open the image and convert to RGB (Three Channels)
    if max_size:
        size = max(max(image.size), max_size)    # Resize the image to the max size
        image = transforms.Resize(size)(image)   # Apply the resize transformation

    if shape:
        image = transforms.Resize(shape)(image)  # Resize the image to the given shape
    
    if transform:
        image = transform(image).unsqueeze(0)    # Apply the transformation and add a batch dimension
    
    '''
    # Un-Transform CODE
    # Remove the batch dimension (squeeze) and convert to a NumPy array for visualization
    image_np = image.squeeze(0).detach().numpy()  
    # If the tensor shape is (C, H, W), we need to transpose it to (H, W, C)
    image_np = image_np.transpose(1, 2, 0)
    # If the image is normalized (e.g., using mean/std), you may need to denormalize it
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_np = (image_np * std) + mean  # Denormalize
    # Clip values to [0, 1] range for display
    image_np = image_np.clip(0, 1)
    # Display the image
    plt.imshow(image_np)
    plt.title('After Transform & Stitched Back Together')
    plt.show()
    '''

    return image.to(device)  # Move the image to the appropriate device

# Create a class VGG, which inherits the base class for all neural network modules
class VGG(nn.Module):

    # Define constructor method
    def __init__(self):

        super(VGG, self).__init__()
        # Required feature layers (Indicates what specific layers we will extract from)
        """
        Target layers from paper :  
                  relu3 3 for the content 
                  relu1 2, relu2 2, relu1 2, relu3 3, relu4 3
        
        Extracting layers 3,8,15,24 corresponding with relu1_2, relu2_2, relu3_3, relu4_3 in paper
        Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (17): ReLU(inplace=True)
        (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (24): ReLU(inplace=True)
        (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (26): ReLU(inplace=True)
        (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (31): ReLU(inplace=True)
        (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (33): ReLU(inplace=True)
        (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (35): ReLU(inplace=True)
        (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        """
        self.req_features = ['3','8','15','24']
        # Load the VGG19 model and keep the first 29 layers
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:29]
        
    # Define the foward pass method
    def forward(self, x):

        features = []  # List to store the features
        # For each layer in the model
        for layer_num, layer in enumerate(self.model):
            x = layer(x)  # Pass the input through each layer

            # IF THE LAYER WE ARE ON IS AN FEATURE-EXTRACTION LAYER
            if str(layer_num) in self.req_features:
                features.append(x)  # Append the features from the required layers

        return features  # Return the list of features

def gram_matrix(input):
    # a     - batch size
    # b     - number of feature maps
    # c,d   - dimensions of feature map
    a,b,c,d = input.size()

    # Resive feature matrix
    features = input.view(a * b, c * d)

    # Compute the gram product
    G = torch.mm(features, features.t())
    
    # Return normalized values by dividing by number of elemens in each feature map
    return G.div(a * b, c * d)

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.targt = gram_matrix(target).detach()
    
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# Normalization layer
# This normalization can be modified if want to use different types of normalization
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        
        # Image tensor will be BxCxHxW square matrix
        # Resize mean and std to C x 1 x 1 Matrix to work with image tensor
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)
        
    def forward(self, img):
        return (img - self.mean) / self.std

# Build model and components
# From the paper - 
# Content Loss : relu3_3 (15)
# Style loss : relu1_2 (3), relu2_2 (8), relu3_3, relu4_3

def build_model(cnn : nn.Module, norm_mean, norm_std, style_img, content_img):
    # Create Normalization layer as first layer
    model = nn.Sequential(Normalization(norm_mean, norm_std))

    for layer in cnn.children():
        # Append loss 

# Function to perform style transfer
def style_transfer(vgg, content, style, iterations=100, lr=0.01):

    # Clone the content image and set requires_grad to True
    # target.required_grad_(True): gradients are computed during backward pass
    # target.to(device): moves the target tensor to our device
    target = content.clone().requires_grad_(True).to(device)

    # Initialize the optimizer
    optimizer = optim.SGD([target], lr=lr)
    
    for i in range(iterations):

        target_features  = vgg(target)  # Get the features of the target image
        content_features = vgg(content) # Get the features of the content image
        style_features   = vgg(style)   # Get the features of the style image
        
        content_loss = calc_content_loss(target_features[1], content_features[1]) # Calculate the content loss
        style_loss   = calc_style_loss(target_features,      style_features)      # Calculate the style loss
        
        # Calculate the total loss
        loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad() # Zero the gradients
        loss.backward()       # Backpropagate the loss
        optimizer.step()      # Update the target image

        # if i % 50 == 0: # Print the loss every 50 iterations
        #    print(f'Iteration {i}, Loss: {loss.item()}')

        if i % 10 == 0:
            print(f'Iteration {i}, Loss: {loss.item()}')
            
            # Print the image every 10 iterations
            output_image = target.cpu().clone().squeeze(0).detach().numpy()  # Remove batch dimension (squeeze), convert to a NumPy array for visualization
            output_image = output_image.transpose(1, 2, 0)  # If the tensor shape is (C, H, W), we need to transpose it to (H, W, C)
            
            # Denormalize the image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            output_image = std * output_image + mean  # Denormalize
            output_image = np.clip(output_image, 0, 1)  # Clip values to [0, 1] range for display
            
            # Convert to PIL image
            output_image = (output_image * 255).astype(np.uint8)  # Convert to uint8
            output_image = Image.fromarray(output_image)  # Convert to PIL image
            
            # Display the image
            plt.imshow(output_image)
            plt.title(f'Iteration {i}')
            plt.axis('off')  # Hide the axis
            
            # Save the image
            output_image.save(f'images/output_image{i}.jpg')
            '''
            output_image = target.cpu().clone().squeeze(0)
            output_image = transforms.ToPILImage()(output_image)
            plt.imshow(output_image)
            plt.title(f'Iteration {i}')
            output_image.save(f'style-transfer/images/output_image{i}.jpg') 
            plt.show() 
            '''     

    
    return target  # Return the target image




# Loss calculation HYPERPARAMETER weights
content_weight = 1.0  # Weight for content loss
style_weight = 100.0   # Weight for style loss

# Call the function to get the device
device = set_device()
print(f"Using device: {device}")

# Load content and style images
content = load_image('content.jpeg', transform)  # Load the content image

''' CONVERSION BACK TO NORMAL IMAGE
# MODIFIED: Convert the tensor back to a PIL image
content_image = content.cpu().clone().squeeze(0).detach().numpy() # Remove batch dimension (squeeze), convert to a NumPy array for visualization
content_image = content_image.transpose(1, 2, 0) # If the tensor shape is (C, H, W), we need to transpose it to (H, W, C)
mean = [0.485, 0.456, 0.406] # If the image is normalized (e.g., using mean/std) ... 
std = [0.229, 0.224, 0.225]  # ... you may need to denormalize it
content_image = (content_image * std) + mean  # Denormalize
content_image = content_image.clip(0, 1) # Clip values to [0, 1] range for display
# content_image = transforms.ToPILImage()(content_image) # Conver to PIL
plt.imshow(content_image)
plt.title('ModiConv | After Load Image')
plt.show()

# ORIGINAL: Convert the tensor back to a PIL image
originalConversion = content.cpu().clone().squeeze(0)  # Move to CPU, clone, and remove batch dimension
originalConversion = transforms.ToPILImage()(originalConversion)  # Convert to PIL image
plt.imshow(originalConversion)
plt.title('OrigConv | After Load Image')
plt.show()
'''

style   = load_image('style.jpeg',   transform, shape=content.shape[-2:])  # Load the style image with the same shape as the style image

# Initialize the model
vgg = VGG().to(device).eval()  # Move the model to the appropriate device and set it to evaluation mode

# Perform style transfer
output = style_transfer(vgg, content, style)  # Perform style transfer

# Save the output image
output_image = output.cpu().clone().squeeze(0)        # Move the output image to the CPU and remove the batch dimension
output_image = transforms.ToPILImage()(output_image)  # Convert the tensor to a PIL (Python Image Library) image
output_image.save('output_image.jpg')                 # Save the output image

# Display the output image
plt.imshow(output_image)
plt.show()

# Modularization: 
# Content Loss Module
# Style Loss Module
# 
"""
Style Loss:
    - How Loss : 
                - Calculated by taking the gram matrix of the current syle features. Then taking the MSE loss between
                  the gram matrix and the current img matrix
    - Why Loss: <Dunno>

Content Loss:
    - How Loss:
               - Calculated by taking gam matrix of the current generated features. Then taking the MSE los between
               the gram matrix and the current img matrix
               
Normalization Module: 
    - Why Normalize : <Dunno>
    - How Normalize :  
                    - In example above we take the MSE loss between the style and generated gram matrices
                    - In example given by prof he takes (img - mean) / std where img matrix is a batch x channel x width x height
                      sized matrix.
    Good Math Explanation : https://stats.stackexchange.com/questions/361723/weight-normalization-technique-used-in-image-style-transfer
"""
