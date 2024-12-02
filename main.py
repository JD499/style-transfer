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
    # Open image
    image = Image.open(img_path).convert("RGB")

    # Resize image, default size 512x512
    image = transforms.Resize(size)(image)

    # Convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Add batch dimension
    image = transform(image).unsqueeze(0)
    return image


# Save processed tensor as image
def save_image(tensor, filename):
    # Clone tensor to CPU and remove batch dimension
    image = tensor.cpu().clone().squeeze(0)

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean

    # Save image to file
    torchvision.utils.save_image(image, filename)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # Load VGG19 model (first 29 layers)
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:29]

        # Replace MaxPool with AvgPool
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.MaxPool2d):
                self.model[i] = nn.AvgPool2d(kernel_size=2, stride=2)

        # Define layers for feature extraction

        # style features (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
        self.style_layers = [0, 5, 10, 19, 28]

        # content features (conv4_2)
        self.content_layer = 21

    def forward(self, x):
        features = []
        content = None

        # Extract features from selected layers
        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            # Collect style features
            if layer_num in self.style_layers:
                features.append(x)

            # Collect content features
            if layer_num == self.content_layer:
                content = x

        return features, content


# Calculate content loss
def calc_content_loss(orig_feature, gen_feature):
    return 0.5 * torch.sum((gen_feature - orig_feature) ** 2)


# Calculate Gram matrix
def gram_matrix(features):
    batch_size, channels, height, width = features.size()
    features = features.view(channels, height * width)
    return torch.mm(features, features.t())


# Calculate style loss
def calc_style_loss(style_features, gen_features):
    loss = 0
    for style, gen in zip(style_features, gen_features):
        # Compute Gram matrices
        style_gram = gram_matrix(style)
        gen_gram = gram_matrix(gen)

        # Get feature dimensions
        batch_size, channels, height, width = gen.size()
        N = channels
        M = height * width

        # Add normalized loss for this layer
        loss += (1.0 / (4 * N**2 * M**2)) * torch.sum((gen_gram - style_gram) ** 2)
    return loss


# Compute total loss (content and style)
def compute_loss(vgg, content, style, generated, content_weight, style_weight):
    # Extract features
    style_features, _ = vgg(style)
    content_features, content_content = vgg(content)
    generated_style, generated_content = vgg(generated)

    # Calculate individual losses
    content_loss = calc_content_loss(content_content, generated_content)
    style_loss = calc_style_loss(style_features, generated_style)

    # Combine losses with weights
    total_loss = (content_weight * content_loss) + (style_weight * style_loss)

    return total_loss


# Style transfer
def style_transfer(
    content_path,
    style_path,
    iterations=300,
    lr=1e-7,
    content_weight=1e-200,
    style_weight=1,
):
    # Setup device
    device = set_device()
    print(f"Using device: {device}")

    # Load images
    content = load_image(content_path).to(device)
    style = load_image(style_path).to(device)

    # Setup VGG
    vgg = VGG().to(device).eval()

    # Clone the content image and set requires_grad to True
    # target.required_grad_(True): gradients are computed during backward pass
    # target.to(device): moves the target tensor to our device
    generated = content.clone().requires_grad_(True).to(device)

    # Initialize optimizer
    optimizer = optim.SGD([generated], lr=lr)

    # Optimization loop
    for i in range(iterations):
        optimizer.zero_grad()

        loss = compute_loss(
            vgg, content, style, generated, content_weight, style_weight
        )

        loss.backward()
        optimizer.step()

        # Print info & image every 50 iterations
        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")

            save_image(generated, f"output_image{i}.jpg")

    return generated


if __name__ == "__main__":
    content_path = "content.jpg"
    style_path = "style.jpg"
    output_path = "output.jpg"

    print("Starting style transfer")

    output = style_transfer(content_path=content_path, style_path=style_path)

    print("Saving output image")
    save_image(output, output_path)
    print(f"Output saved as {output_path}")
