"""Quick visual check of preprocessing output."""
import random

import cv2
from pyprojroot import here

output_dir = here("data/preprocessed")
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

# Spot-check one label file
label = here("data/preprocessed/yolo/labels/train").glob("*.txt").__next__()
content = label.read_text().strip()
parts = content.split()
print(f"Class: {parts[0]}")  # should be "0"
print(f"Points: {(len(parts) - 1) // 2}")  # should be > 10
# All coordinate values should be between 0 and 1
coords = [float(x) for x in parts[1:]]
print(f"Coord range: {min(coords):.4f} – {max(coords):.4f}")  # should be 0.0–1.0