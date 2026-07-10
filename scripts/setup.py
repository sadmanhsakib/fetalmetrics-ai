import preprocess

from dataset import fetch_dataset


if __name__ == "__main__":
    fetch_dataset()

    preprocess.preprocess(
        csv_path=preprocess.DATASET_PATH,
        images_dir=preprocess.IMAGE_DIR,
        output_dir=preprocess.OUTPUT_DIR,
        val_split=0.15,
    )  
    print("✅ Setup completed.")
    print("To Run the webapp, run the following command:")
    print("uv run streamlit run src/app.py")