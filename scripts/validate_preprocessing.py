"""Quick visual check of preprocessing output."""
import cv2
import numpy as np
from pathlib import Path
import random

output_dir = Path("data/preprocessed")
img_dir = output_dir / "fastai" / "images" / "train"
mask_dir = output_dir / "fastai" / "masks" / "train"

# Pick 3 random samples
train_files = list(img_dir.glob("*.png"))
samples = random.sample(train_files, min(3, len(train_files)))

for img_path in samples:
    mask_path = mask_dir / img_path.name
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Overlay mask on image
    overlay = img.copy()
    overlay[mask > 0] = [0, 255, 0]  # green overlay
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # Print stats
    coverage = (mask > 0).sum() / mask.size * 100
    print(f"{img_path.name}: mask coverage = {coverage:.1f}%") # should be 20-40%

    cv2.imshow(f"Check: {img_path.name}", blended)

cv2.waitKey(0)
cv2.destroyAllWindows()