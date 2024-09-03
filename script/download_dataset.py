import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up the Kaggle API with the credentials
api = KaggleApi()
api.authenticate()

# Specify the dataset and download path
dataset = 'techsash/waste-classification-data'
path = '.'

# Download the dataset
api.dataset_download_files(dataset, path=path, unzip=True)
