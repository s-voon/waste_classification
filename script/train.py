import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from src.data_preparation import prepare_data
from src.trainer import trainer

def main(train_dir, test_dir,num_epochs=10, valid_size=0.2, batch_size=64, image_size=(64, 64)):
    # Prepare data
    train_loader, valid_loader, _, _ = prepare_data(train_dir, test_dir,valid_size=valid_size, batch_size=batch_size, image_size=image_size)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load a pre-trained ResNet34 model
    model = models.resnet34(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 250),
        nn.ReLU(),
        nn.Linear(250, 2)
    )
    
    # Move the model to the available device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    trainer(model, criterion, optimizer, train_loader, valid_loader, device, epochs=num_epochs)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/model_waste.pt')
    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script/train.py <train_dir> <test_dir> <num_epochs>")
        sys.exit(1)

    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    num_epochs = int(sys.argv[3])

    main(train_dir, test_dir, num_epochs)