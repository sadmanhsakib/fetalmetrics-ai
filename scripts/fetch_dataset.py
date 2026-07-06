import os
import shutil

import kagglehub

destination_path = "data/"

# removing the old dir for recency
if os.path.exists(destination_path):
    shutil.rmtree(destination_path)

os.makedirs(destination_path, exist_ok=True)

# Download the dataset
path = kagglehub.dataset_download("thanhbnhphan/hc18-grand-challenge")

# move the dataset
shutil.move(path, destination_path)

os.rename(os.path.join(destination_path, os.path.basename(path)),
          os.path.join(destination_path, "raw"))

print(f"✅ Dataset stored in {destination_path}/raw")