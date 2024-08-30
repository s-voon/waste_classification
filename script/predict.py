import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def main(image_path):
    # Load the pre-trained ResNet34 model
    model = models.resnet34(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 250),
        torch.nn.ReLU(),
        torch.nn.Linear(250, 2)
    )
    
    # Load the model weights
    model.load_state_dict(torch.load('model_waste.pt'))
    model.eval()  # Set the model to evaluation mode
    
    # Define image transformations
    IMAGE_SIZE = 64
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    
    # Make prediction
    with torch.no_grad():  # Disable gradient calculation
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # Interpret the prediction
    classes = ['Organic', 'Recyclable']
    predicted_class = classes[predicted.item()]
    
    print(f'Predicted class: {predicted_class}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make a prediction with a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    args = parser.parse_args()
    
    main(args.image_path)
