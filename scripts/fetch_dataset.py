import os

import kagglehub

download_path = "data/raw/"
os.makedirs(download_path, exist_ok=True)

# Download the dataset
path = kagglehub.dataset_download("thanhbnhphan/hc18-grand-challenge",
                                  path=download_path)

print("Path to dataset files:", path)