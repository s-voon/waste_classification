import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision import models
import argparse


def main(model_path, test_data_path):
    # Load the model
    # Load a pre-trained resnet34
    model = models.resnet34(weights='DEFAULT')

    # Freeze all layers parameter
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
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Track test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # Define class labels
    classes = ['O', 'R']

    IMAGE_SIZE = (64, 64)
    # Define test data loader
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Iterate over test data
    for data, target in test_loader:
        # Move tensors to GPU if CUDA is available
        data, target = data.to(device), target.to(device)

        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # Calculate the batch loss
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)

        # Convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # Compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy()) if device == torch.device('cuda') else np.squeeze(correct_tensor.numpy())

        # Calculate test accuracy for each object class
        for i in range(len(classes)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # Average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    # Print test accuracy for each class
    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                int(class_correct[i]), int(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    # Print overall test accuracy
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        int(np.sum(class_correct)), int(np.sum(class_total))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model.')
    parser.add_argument('model_path', type=str, help='Path to the saved model.')
    parser.add_argument('test_data_path', type=str, help='Path to the test data directory.')
    args = parser.parse_args()

    main(args.model_path, args.test_data_path)
