import os
import shutil

import kagglehub

destination_path = "data/"


def fetch_dataset():
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


def upload_dataset(handle, local_dataset_dir, is_new: bool = False):
    if is_new:
        kagglehub.dataset_upload(handle, local_dataset_dir)
    else:
        version_notes = input("Enter version notes: ")
        kagglehub.dataset_upload(handle, local_dataset_dir, version_notes=version_notes)


if __name__ == "__main__":
    fetch_dataset()