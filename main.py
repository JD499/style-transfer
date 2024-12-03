import cv2
import numpy as np
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
    lr=8000,
    content_weight=1e-60,
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
    optimizer = optim.SGD([generated], lr=lr, momentum=0.9)

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


# Convert the youtube EAC format to equirectangular
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


def get_frame_from_video(video_path, frame_number=0, output_path="frame.jpg"):
    # Open video file
    video = cv2.VideoCapture(video_path)

    # Go to frame
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read frame
    success, frame = video.read()
    if not success:
        print("Failed to extract frame")
        return output_path

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame from EAC to equirectangular
    equirect_frame = convert_eac_equirectangular(frame)

    # Save the converted frame
    Image.fromarray(equirect_frame).save(output_path)
    print(f"Converted frame saved as {output_path}")

    video.release()
    return output_path


if __name__ == "__main__":
    process_video = True
    frame_number = 100

    # Input paths
    content_path = "content.jpg"
    style_path = "style.jpg"
    output_path = "output.jpg"
    video_path = "360video.mp4"

    # Process video
    if process_video:
        print("Processing video frame...")
        content_path = get_frame_from_video(
            video_path, frame_number=frame_number, output_path="content.jpg"
        )

    # Style transfer
    print("Starting style transfer")
    output = style_transfer(content_path=content_path, style_path=style_path)

    # Save result image
    print("Saving output image")
    save_image(output, output_path)
    print(f"Output saved as {output_path}")

    # TODO: Add more optimizers
    # TODO: Make a more consistent config path/description
    # TODO: Use config to cycle learning rates, weights, optimizers
    # TODO: Add more console out to describe what is happening
    # TODO: Generate more plots/visualizations
    # TODO: Make the images show more growth between them (Loss stuff)

    '''
    2.  Implement a deep learning system, train, validate and test it. This is the part that will
        take most of the time. You may explore various neural network architectures (e.g., MLP, CNN, LSTM,
        RNN). You will do a hyper-parameter search, which includes but is not limited to: network size
        (number of layers, number of nodes per layer), activation functions, learning rate, batch size, number
        of epochs, optimizer, loss function, dropout rate, regularization method, random restarts, data split
        etc. I encourage you to use PyTorch, but you are welcome to use any other framework.

    3.  Write a report describing what you did. The report is a mini research paper. It contains title,
        author names, abstract, and several sections that cover: introduction to the problem, what you set
        out to achieve, methodology, experimental setup, explanation of results, conclusion that summarizes
        what we learn from your work. Include useful plots of the data you collected during training (e.g.
        accuracy, loss, convergence time, memory usage etc), images, diagrams or other visuals that facilitate
        your exposition. Do not include too much code, unless it is really necessary.
    '''
