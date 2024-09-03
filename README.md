# Efficient Waste Classification with ResNet34: Leveraging Transfer Learning for Accurate Results

![Recycling bin](recycle.png)

## Introduction
Waste management is a critical global issue, affecting environmental sustainability, public health, and resource conservation. As urban populations grow and consumption patterns change, efficient waste sorting and recycling become increasingly important to minimize landfill waste and maximize resource recovery.

The `Efficient Waste Classification with ResNet34` project addresses this challenge by harnessing the power of advanced computer vision techniques and transfer learning. By leveraging a pre-trained ResNet34 model, this solution enables accurate and efficient classification of waste materials into distinct categories, such as organic and recyclable. This automated approach not only enhances sorting precision but also streamlines waste management processes.

## Overview
In this project, we use a pre-trained ResNet34 model, originally trained on a large dataset, and fine-tune it for our specific image classification task. We modify the final classification layer to suit binary classification and evaluate the model's performance on a test dataset. This approach leverages the model's pre-learned features to efficiently classify new images.

## Features
- **Pre-trained Model:** Utilize ResNet34, a deep residual network known for its accuracy and efficiency.
- **Custom Classification:** Adapt the model for binary classification by replacing the final layer.
- **Inference Script:** A ready-to-run script for fine-tuning the per-trained model, evaluating the model, and making predictions on new images.

## Installation

1.  **Clone the repository**: Open the terminal and run the following command:

    ```         
    git clone git@github.com:s-voon/waste_classification.git
    ```

    Navigate to the directory of the cloned repository.

2.  **Create environment**: Navigate to the directory where the environment.yaml file is located on your terminal and run the following command:

    ```         
    conda env create -f environment.yaml
    conda activate waste_org
    ```

3. **Dataset download**: The dataset used in this repository is sourced from Kaggle. You can find it [here](https://www.kaggle.com/datasets/techsash/waste-classification-data).

    To download the dataset, follow these steps:

    - Set Up Kaggle API: Follow the instructions provided in the [Kaggle API documentation](https://www.kaggle.com/docs/api) to set up your Kaggle API credentials.

    - Download the Dataset:  After setting up your API credentials, run the following command to download the dataset:
    ```bash
    python script/download_dataset.py
    ```

## Usage

For detailed, step-by-step instructions on how to train the model and use it for inference, please refer to the [notebook](notebook/waste_classification.ipynb) directory.

- **Fine tune a Pre-Trained (ResNet34) model:**
    Before fine-tuning, ensure the dataset is downloaded as per the instructions above. Then, to fine-tune a pre-trained model, run:
    ```bash
    python script/train.py "DATASET/TRAIN/"  20
    ```
    Replace `DATASET/TRAIN/` with the paths to your training datasets. The last argument (10) indicates the number of epochs or any specific parameter required for training.

- **Evaluate the Model:** To evaluate the trained model on the test dataset, use the following command:
    ```bash
    python script/evaluate.py models/model_waste.pt "DATASET/TEST/"
    ```
    Replace `model_waste.pt` with the path to your saved model file, and `DATASET/TEST/` with the path to your test data.

- **Make Predictions:** To make predictions using the trained model from the terminal, run the following command:
    ```bash
    python script/predict.py organic_sample.jpeg
    ```    

## Contributing Guidelines

Interested in contributing? Check out the [contributing guidelines](CONTRIBUTING.md). Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## License

The project is licensed under the terms of the [MIT license](https://github.com/s-voon/waste_classification/blob/main/LICENSE).
