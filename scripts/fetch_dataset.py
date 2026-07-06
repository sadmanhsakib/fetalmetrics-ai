import os
import shutil

import kagglehub
from pyprojroot import here

download_path = "data/raw/"
os.makedirs(download_path, exist_ok=True)

# Download the dataset
path = kagglehub.dataset_download("thanhbnhphan/hc18-grand-challenge")
# move the dataset
shutil.move(path, here(download_path))

print("Path to dataset files:", path)