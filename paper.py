# Import required packages
from collections import namedtuple
import time
import torch
import os
import cv2
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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

# Struggle #2
# First model implementation used three modules where the style and feature loss were seperate and acted as seperate modules
# joined in another module that computed the perceptual loss. The way that this was implemented required the retain_graph option
# to be set to True as well as repeated tensor cloning to avoid inplace operations. When doing this the entire computation graph 
# along with the cloned tensors (larges ones around 64,60000 for the gram matrices) started to eat GPU memory and within 20 iterations 
# broke 20gb of allocated memory which forced the training to stop on a 4060 limited to 16gb of available memory. 

# Struggle #3
# We find that there are strange artifacts and cross-hatching that appears throughout
# We think that one way to deal with this is to implement the suggestion presented in Ruder 
# where instead of the maxpooling used in the VGG19 model we instead replace that with
# an avgpool layer which should (FIGURE OUT WHY) create more smooth styling. One of the things that we 
# notice with the style is that finer style details, like the starry nights swirl effects are being lost
# We also work on fairly low resolution images so we have to adapt the network, prob using something presented
# in the paper to downsample the image in a learned manner and apply the training on patches.


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
            # nn.ReflectionPad2d(2),
            # Paper specifies that fraction tranposes are used 
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=(9,9),
                      stride=2),
            nn.InstanceNorm2d(32, affine=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.ReflectionPad2d(2),
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
        print('output {}'.format(output.size()))
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
            nn.ReflectionPad2d(2),
            nn.ConvTranspose2d(in_channels=64,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=2),
            nn.InstanceNorm2d(32, affine=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d(2),
            nn.ConvTranspose2d(in_channels=32,
                      out_channels=3,
                      kernel_size=(9,9),
                      stride= 2,
                      padding=4,
                      output_padding=1),
            nn.InstanceNorm2d(3, affine=True), 
            nn.Tanh()
        )
    
    def forward(self, x):
        output = self.upblock(x)
        print('upsamp {}'.format(output[:,:,:296,:296].size()))
        return output[:,:,:296,:296]

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
        
# Total Variance Regularization : (TRY TO FIND THE PURPOSE OF THIS TO EXPLAIN IN THE PAPER)
def tvr(x):    
    return torch.sum(torch.abs(x[:,:,:,:-1]-x[:,:,:,1:])) + torch.sum(torch.abs(x[:,:,:-1,:]-x[:,:,1:,:]))

# Output format is to prevent any inplace calculations
LossOutput = namedtuple(
    "LossOutput", ['relu1_1' ,'relu2_1' ,'relu3_1' ,'relu4_1' ,'relu5_1' ,'relu4_2'
]
)
class LossNetwork(nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        # Only intersted in a certain set of features 3,8,15,24
        self.layer_mapping = {
            # Style Loss
            '1' : 'relu1_1',
            '6' : 'relu2_1',
            '11': 'relu3_1',
            '20': 'relu4_1',
            '29': 'relu5_1',
            # Content Loss
            '22': 'relu4_2'
        }
        self.model = models.vgg.vgg19(pretrained=True).features[:30]

        # Make these replacements on the maxpool layers to isntead 
        # maintain a smoother output appearance
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.MaxPool2d):
                self.model[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        output = {}
        
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layer_mapping.keys():
                output[self.layer_mapping[name]] = x
                
        return LossOutput(**output)
        
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

def convert_eac_equirectangular(frame):
    # Get dimensions of frame
    height, width = frame.shape[:2]

    # Calculate dimensions of each cube face
    face_width = width // 3  # Each face is 1/3 of the width
    face_height = height // 2  # Each face is 1/2 of the height

    # Extract cube faces from the frame
    # - EAC format is a 3x2 matrix
    # - Layout: [Left, Front, Right]
    #          [Down, Back, Up]
    faces = {
        "left": frame[0:face_height, 0:face_width],
        "front": frame[0:face_height, face_width : face_width * 2],
        "right": frame[0:face_height, face_width * 2 : width],
        "down": cv2.rotate(
            frame[face_height:height, 0:face_width], cv2.ROTATE_90_CLOCKWISE
        ),
        "back": cv2.rotate(
            frame[face_height:height, face_width : face_width * 2],
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ),
        "up": cv2.rotate(
            frame[face_height:height, face_width * 2 : width], cv2.ROTATE_90_CLOCKWISE
        ),
    }

    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Spherical coordinates
    # phi: longitude angle (-π to π)
    # theta: latitude angle (π/2 to -π/2)
    phi = np.linspace(-np.pi, np.pi, width)
    theta = np.linspace(np.pi / 2, -np.pi / 2, height)
    phi, theta = np.meshgrid(phi, theta)

    # Convert spherical coordinates to cartesian coordinates
    # x = cos(theta) * sin(phi)
    # y = sin(theta)
    # z = cos(theta) * cos(phi)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta)
    z = np.cos(theta) * np.cos(phi)

    # Map each equirectangular pixel to corresponding cube face
    for i in range(height):
        for j in range(width):
            # Get absolute values to determine which face to use
            abs_x, abs_y, abs_z = abs(x[i, j]), abs(y[i, j]), abs(z[i, j])
            max_val = max(abs_x, abs_y, abs_z)

            # Map pixel to cube face based on largest coordinate
            if abs_x == max_val:
                # Right or left face
                if x[i, j] > 0:
                    face = faces["right"]
                    u = (-z[i, j] / abs_x + 1) * face.shape[1] / 2
                    v = (-y[i, j] / abs_x + 1) * face.shape[0] / 2
                else:
                    face = faces["left"]
                    u = (z[i, j] / abs_x + 1) * face.shape[1] / 2
                    v = (-y[i, j] / abs_x + 1) * face.shape[0] / 2
            elif abs_y == max_val:
                # Up or down face
                if y[i, j] > 0:
                    face = faces["up"]
                    u = (x[i, j] / abs_y + 1) * face.shape[1] / 2
                    v = (z[i, j] / abs_y + 1) * face.shape[0] / 2
                else:
                    face = faces["down"]
                    u = (x[i, j] / abs_y + 1) * face.shape[1] / 2
                    v = (-z[i, j] / abs_y + 1) * face.shape[0] / 2
            else:
                # Front or back face
                if z[i, j] > 0:
                    face = faces["front"]
                    u = (x[i, j] / abs_z + 1) * face.shape[1] / 2
                    v = (-y[i, j] / abs_z + 1) * face.shape[0] / 2
                else:
                    face = faces["back"]
                    u = (-x[i, j] / abs_z + 1) * face.shape[1] / 2
                    v = (-y[i, j] / abs_z + 1) * face.shape[0] / 2

            u = min(max(0, int(u)), face.shape[1] - 1)
            v = min(max(0, int(v)), face.shape[0] - 1)

            # Assign to equirectangular image
            frame[i, j] = face[v, u]

    return frame

def frobenius_norm(pred, target):
    return torch.linalg.matrix_norm(pred - target, ord='fro').pow(2).mean()

def train(model_output,
          debug_output,
          data_source,
          style_source,
          iterations,
          style_weight = 1.0,
          content_weight = 1.0,
          tvr_weight = 1.0,
          debug_interval = 100, 
          epochs = 1, 
          batch_size = 4, 
          lr = 0.001, 
          lr_growth = 1.0,
          cuda_limit = 1.0) -> None:
    """Trains our style transfer model. Upon training a model.pt file is outputed at a desired directory

    Args:
        model (nn.Module): style transfer model to train
        loss_network (nn.Module): Loss model to optimize
        model_output (PathLike): Model.pt output location
        debug_output (PathLike): Location to output the debug images
        data_source (PathLike): Location of training data. Ensure data is formated in a manner that the DataLoader can process
        style_source (PathLike): Location of style photo to train on
        iteartions (int): Number of iterations per epoch
        style_weight (PathLike): Location of training data. Ensure data is formated in a manner that the DataLoader can process
        content_weight (PathLike): Location of training data. Ensure data is formated in a manner that the DataLoader can process
        tvr_weight (PathLike): Location of training data. Ensure data is formated in a manner that the DataLoader can process
        debug_interval (PathLike): Interval to log training information and save image to debug_output
        epochs (int): # of desired epochs
        batch_size (int): Size of training batch
        lr (float): Intial optimizer learning rate
        lr_growth (float): Growth factor which augments the learning rate per epoch 
        cuda_limit (float): Cuda per process memory usage fraction between [0,1] 
    """
    # Necessary hard parameters
    IMAGE_SIZE = 256
    BATCH_SIZE = batch_size
    DEBUG_INTERVAL = debug_interval
    EPOCHS = epochs
    LR = lr
    LR_GROWTH = lr_growth
    
    # Path validation
    print('Validating paths')
    if not os.path.exists(style_source):
        raise ValueError('Style path does not exist')
    STYLE_SOURCE = style_source
    
    if not os.path.exists(data_source):
        raise ValueError('Training data source does not exist')
    DATA_SOURCE = data_source
    
    if not os.path.exists(data_source):
        raise ValueError('Training data source does not exist')
    DATA_SOURCE = data_source
    
    if not os.path.exists(model_output):
        MODEL_OUTPUT = os.getcwd()
    else: 
        MODEL_OUTPUT = model_output

    if not os.path.exists(debug_output):
        DEBUG_OUTPUT = os.getcwd()
    else: 
        DEBUG_OUTPUT = debug_output

    # 0.5 1e5 1e-5
    CONTENT_WEIGHT = content_weight
    STYLE_WEIGHT = style_weight
    TVR_WEIGHT = tvr_weight
 
    # CUDA parameter setup
    print('Setting up CUDA parameters')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (DEVICE.type == "cuda"):
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(cuda_limit)
        
    # Training parameter setup
    torch.autograd.set_detect_anomaly(True)
    torch.set_default_tensor_type('torch.FloatTensor')

    print('Loading data')
    # Training data setup
    # transform = EquirecTransform(IMAGE_SIZE)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        nn.CircularPad2d(20),
        transforms.Normalize(
        # mean and std taken from ImageNet norm & std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(DATA_SOURCE, transform)
    train_loader = DataLoader(train_data, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=0)

    print('Initializing loss network')
    # Initialize loss network
    with torch.no_grad():
        loss_network = LossNetwork()
        loss_network.to(DEVICE)
        mse = torch.nn.MSELoss()
    loss_network.eval()
    
    # prepare style tensor
    print('Preparing style tensor')
    style_image = Image.open(STYLE_SOURCE).convert('RGB')
    with torch.no_grad():
        style_img_tensor = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
            # mean and std taken from ImageNet norm & std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])(style_image).unsqueeze(0)
        style_img_tensor = style_img_tensor.to(DEVICE)
    print(style_img_tensor.size())
    
    # Model setup
    print('Setting up model')
    model = StyleTransferModel()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)

    # Preload style Gram Matrix
    with torch.no_grad():
        style_feature_loss = loss_network(style_img_tensor)
        style_gram = [gram_matrix(y) for y in style_feature_loss[:5]]
    
    # Initialize loss tracking
    feature_loss_tracker = []
    style_loss_tracker = []
    tvr_loss_tracker = []
    total_loss_tracker = []
    
    total_feature_loss = 0
    total_style_loss = 0
    total_tvr_loss = 0
    total_loss = 0
    
    # Intialize Data Capture
    writer = SummaryWriter()
    
    # Initiate training loop
    count = 0
    curr_epoch = 0
    optimizer.zero_grad()
    while curr_epoch < EPOCHS:
        for x, _ in train_loader:
            count += 1
            print(count)
            
            feature_loss = 0
            style_loss = 0
            tvr_loss = 0
            
            # Pass image through model
            x = x.to(DEVICE)
            y = model(x)
            
            # Isolate Conent
            with torch.no_grad():
                x_feature = x.detach()
            
            # Calculate losses
            y_loss = loss_network(y)
            x_loss = loss_network(x_feature)
            
            print(y.size())
            print(x_feature.size())
            with torch.no_grad():
                # Grabbing loss for content
                x_feature_loss = x_loss[5].detach()
          
            print(y_loss[5].size())
            print(x_feature_loss.size())
            feature_loss = CONTENT_WEIGHT * mse(y_loss[5], x_feature_loss)

            for i in range(0,4):
                print(y_loss[i].size())
                y_gram = gram_matrix(y_loss[i])
                style_loss += frobenius_norm(y_gram, style_gram[i].expand_as(y_gram))
            style_loss = STYLE_WEIGHT * style_loss
            
            tvr_loss = TVR_WEIGHT * tvr(y)
            
            # UPdate total loss
            total_loss = feature_loss + style_loss + tvr_loss
            total_loss_tracker.append(total_loss)

            # Update trackers
            feature_loss_tracker.append(feature_loss)
            style_loss_tracker.append(style_loss)
            tvr_loss_tracker.append(tvr_loss)
            
            total_feature_loss += feature_loss
            total_style_loss += style_loss
            total_tvr_loss += tvr_loss
            
            total_loss.backward()
            optimizer.step()
            
            # DEBUG INTERVAL
            if count % DEBUG_INTERVAL == 0:
                # Update summary writer with info
                actual_count = count * (curr_epoch + 1)
                writer.add_scalar('Loss/Individual/content', feature_loss, actual_count)
                writer.add_scalar('Loss/Individual/style', style_loss, actual_count)
                writer.add_scalar('Loss/Individual/tvr', tvr_loss, actual_count)
                writer.add_scalar('Loss/Total/content', total_feature_loss, actual_count)
                writer.add_scalar('Loss/Total/style', total_style_loss, actual_count)
                writer.add_scalar('Loss/Total/tvr', total_tvr_loss, actual_count)
                
                # Print debug log
                mesg = "{} [{}] LR: {} content: {:.2f}  style: {:.2f}  reg: {:.2f} total: {:.6f}".format(
                            time.ctime(), count, LR,
                            total_feature_loss / DEBUG_INTERVAL,
                            total_style_loss / DEBUG_INTERVAL,
                            total_tvr_loss / DEBUG_INTERVAL,
                            (total_loss ) / DEBUG_INTERVAL
                        )
                print(mesg)
                total_feature_loss = 0
                total_style_loss = 0
                total_tvr_loss = 0
                model.eval()
                y = model(x)
                save_debug_image(x,y.detach(), os.path.join(DEBUG_OUTPUT, 'debug_{}_{}.png'.format(count, curr_epoch)))
                model.train()
                
            # Epoch count
            if count >= iterations:
                curr_epoch += 1
                count = 0
                break

        print('Moved through loaded_data')
        # Model learn rate updated per epoch
        LR = LR * LR_GROWTH
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)            

    torch.save(model, os.path.join(MODEL_OUTPUT, 'model_test.pt'))
    writer.close()
     
if __name__ == "__main__":
    train(
        model_output='./models',
        debug_output='./debug',
        data_source='./train',
        style_source='style.jpg',
        epochs=2,
        content_weight=0.5,
        style_weight=1e5,
        tvr_weight=1e-5,
        iterations = 5,
        debug_interval=1,
        cuda_limit=0.65
    )