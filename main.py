import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torchvision import models, transforms
from torchvision.models import VGG19_Weights


# Set device as Metal, CUDA, or CPU
def set_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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


def save_image(tensor, filename):
    # Clone the tensor to CPU
    image = tensor.cpu().clone().squeeze(0)

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean

    # Save image
    torchvision.utils.save_image(image, filename)


# Create a class VGG, which inherits the base class for all neural network modules
class VGG(nn.Module):
    # Define constructor method
    def __init__(self):
        super(VGG, self).__init__()

        # Load first 29 layers of VGG19 Model
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:29]

        # Replace MaxPool with AvgPool
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.MaxPool2d):
                self.model[i] = nn.AvgPool2d(kernel_size=2, stride=2)

        # VGG19 layers for style features (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
        self.style_layers = [0, 5, 10, 19, 28]

        # VGG19 layer for content features (conv4_2)
        self.content_layer = 21

    # Define the foward pass method
    def forward(self, x):
        features = []  # List to store the features
        content = None

        # For each layer in the model
        for layer_num, layer in enumerate(self.model):
            x = layer(x)  # Pass the input through each layer

            # IF THE LAYER WE ARE ON IS AN FEATURE-EXTRACTION LAYER
            if layer_num in self.style_layers:
                features.append(x)  # Append the features from the required layers

            if layer_num == self.content_layer:
                content = x

        return features, content  # Return the list of features


# Content Loss
def calc_content_loss(orig_feature, gen_feature):
    return 0.5 * torch.sum((gen_feature - orig_feature) ** 2)


def gram_matrix(features):
    batch_size, channels, height, width = features.size()
    features = features.view(channels, height * width)
    return torch.mm(features, features.t())


# Style Loss
def calc_style_loss(style_features, gen_features):
    loss = 0
    for style, gen in zip(style_features, gen_features):
        style_gram = gram_matrix(style)
        gen_gram = gram_matrix(gen)

        batch_size, channels, height, width = gen.size()
        N = channels
        M = height * width

        loss += (1.0 / (4 * N**2 * M**2)) * torch.sum((gen_gram - style_gram) ** 2)
    return loss


# Function to perform style transfer
def style_transfer(
    content_path,
    style_path,
    iterations=300,
    lr=1e-7,
    content_weight=1e-200,
    style_weight=1,
):
    device = set_device()
    print(f"Using device: {device}")

    # Load images
    content = load_image(content_path).to(device)
    style = load_image(style_path).to(device)

    vgg = VGG().to(device).eval()

    # Clone the content image and set requires_grad to True
    # target.required_grad_(True): gradients are computed during backward pass
    # target.to(device): moves the target tensor to our device
    generated = content.clone().requires_grad_(True).to(device)

    # Initialize the optimizer
    optimizer = optim.SGD([generated], lr=lr)

    for i in range(iterations):
        generated_style, generated_content = vgg(
            generated
        )  # Get the features of the target image
        content_features, content_content = vgg(
            content
        )  # Get the features of the content image
        style_features, _ = vgg(style)  # Get the features of the style image

        content_loss = calc_content_loss(content_content, generated_content)
        style_loss = calc_style_loss(style_features, generated_style)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()  # Zero the gradients
        total_loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the target image

        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {total_loss.item()}")

            save_image(generated, f"output_image{i}.jpg")

    return generated  # Return the target image


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
