import time
import pandas as pd
from pathlib import Path


csv_path = Path("data/raw/training_set_pixel_size_and_HC.csv")
img_dir = Path("data/raw/training_set")


def main():
    validate_dataset()
    
    
def validate_dataset():
    df = pd.read_csv(csv_path)
    print(f"CSV rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst row:\n{df.iloc[0]}")
    print(f"\nImages found: {len(list(img_dir.glob('*.png')))}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nPixel size range: {df['pixel size(mm)'].min():.4f} – {df['pixel size(mm)'].max():.4f} mm")
    print(f"HC range: {df['head circumference (mm)'].min():.1f} – {df['head circumference (mm)'].max():.1f} mm")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"✅ Execution completed in {time.time() - start_time:.2f} seconds")
