import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms
from torchvision.models import VGG19_Weights


# Assign user's device for optimal runtime (GPU or CPU)
def set_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Use MPS (Metal Performance Shaders) if available
    elif torch.cuda.is_available():
        return torch.device("cuda")  # Use CUDA (GPU) if available
    else:
        return torch.device("cpu")  # Otherwise, use CPU


# Load and preprocess image
def load_image(img_path, size=(512, 512)):
    # Open Image
    image = Image.open(img_path).convert("RGB")

    # Resize Image default value 512x512
    image = transforms.Resize(size)(image)

    # Convert to tensor and Normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Add dimension at index 0 for batch size
    image = transform(image).unsqueeze(0)

    """
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
    """

    return image


def save_image(tensor, output_path):
    # Convert back to PIL Image and save
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(output_path)

    # Display the output image
    plt.imshow(image)  # Display the output image
    plt.show()  # Show the plot


# Create a class VGG, which inherits the base class for all neural network modules
class VGG(nn.Module):
    # Define constructor method
    def __init__(self):
        super(VGG, self).__init__()
        # Required feature layers (Indicates what specific layers we will extract from)
        self.req_features = ["0", "5", "10", "19", "28"]
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


# Function to define loss calculation
def calc_content_loss(gen_feature, orig_feature):
    # Content loss calculated by adding MSE loss between content and generated feature then added
    content_l = torch.mean(
        (gen_feature - orig_feature) ** 2
    )  # Calculate the mean squared error loss
    return content_l  # Return the content loss


# Function to calculate style loss
def calc_style_loss(gen_features, style_features):
    style_l = 0
    for gen, style in zip(gen_features, style_features):
        _, c, h, w = gen.size()  # Get the dimensions of the generated features
        gen = gen.view(c, h * w)  # Reshape the generated features
        style = style.view(c, h * w)  # Reshape the style features
        gen = torch.mm(
            gen, gen.t()
        )  # Calculate the Gram matrix for the generated features
        style = torch.mm(
            style, style.t()
        )  # Calculate the Gram matrix for the style features
        style_l += torch.mean((gen - style) ** 2) / (
            c * h * w
        )  # Calculate the mean squared error loss and normalize
    return style_l  # Return the style loss


# Function to perform style transfer
def style_transfer(content_path, style_path, iterations=300, lr=0.01):
    device = set_device()
    print(f"Using device: {device}")

    # Load images
    content = load_image(content_path).to(device)
    style = load_image(style_path).to(device)

    vgg = VGG().to(device).eval()

    # Clone the content image and set requires_grad to True
    # target.required_grad_(True): gradients are computed during backward pass
    # target.to(device): moves the target tensor to our device
    target = content.clone().requires_grad_(True).to(device)

    # Initialize the optimizer
    optimizer = optim.SGD([target], lr=lr)

    for i in range(iterations):
        target_features = vgg(target)  # Get the features of the target image
        content_features = vgg(content)  # Get the features of the content image
        style_features = vgg(style)  # Get the features of the style image

        content_loss = calc_content_loss(
            target_features[1], content_features[1]
        )  # Calculate the content loss
        style_loss = calc_style_loss(
            target_features, style_features
        )  # Calculate the style loss

        # Calculate the total loss
        loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the target image

        # if i % 50 == 0: # Print the loss every 50 iterations
        #    print(f'Iteration {i}, Loss: {loss.item()}')

        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")

            # Print the image every 10 iterations
            output_image = (
                target.cpu().clone().squeeze(0).detach().numpy()
            )  # Remove batch dimension (squeeze), convert to a NumPy array for visualization
            output_image = output_image.transpose(
                1, 2, 0
            )  # If the tensor shape is (C, H, W), we need to transpose it to (H, W, C)

            # Denormalize the image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            output_image = std * output_image + mean  # Denormalize
            output_image = np.clip(
                output_image, 0, 1
            )  # Clip values to [0, 1] range for display

            # Convert to PIL image
            output_image = (output_image * 255).astype(np.uint8)  # Convert to uint8
            output_image = Image.fromarray(output_image)  # Convert to PIL image

            # Display the image
            plt.imshow(output_image)
            plt.title(f"Iteration {i}")
            plt.axis("off")  # Hide the axis

            # Save the image
            output_image.save(f"output_image{i}.jpg")
            """
            output_image = target.cpu().clone().squeeze(0)
            output_image = transforms.ToPILImage()(output_image)
            plt.imshow(output_image)
            plt.title(f'Iteration {i}')
            output_image.save(f'style-transfer/images/output_image{i}.jpg') 
            plt.show() 
            """

    return target  # Return the target image


# Loss calculation HYPERPARAMETER weights
alpha = 1.0  # Weight for content loss
beta = 1.0  # Weight for style loss


""" CONVERSION BACK TO NORMAL IMAGE
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
"""


if __name__ == "__main__":
    content_path = "content.jpg"
    style_path = "style.jpg"
    output_path = "output.jpg"

    print("Starting style transfer")

    # Run style transfer
    output = style_transfer(
        content_path=content_path,
        style_path=style_path,
    )

    # Save result
    print("Saving output image")
    save_image(output, output_path)
    print(f"Output saved as {output_path}")
