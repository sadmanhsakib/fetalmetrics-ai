"""Quick visual check of preprocessing output."""
import random

import cv2
from pyprojroot import here

OUTPUT_DIR = here("data/preprocessed")
IMG_DIR = OUTPUT_DIR / "fastai" / "images" / "train"
MASK_DIR = OUTPUT_DIR / "fastai" / "masks" / "train"

# Pick 3 random samples
train_files = list(IMG_DIR.glob("*.png"))
samples = random.sample(train_files, min(3, len(train_files)))

for img_path in samples:
    mask_path = MASK_DIR / img_path.name
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Overlay mask on image
    overlay = img.copy()
    overlay[mask > 0] = [0, 255, 0]  # green overlay
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # Print stats
    coverage = (mask > 0).sum() / mask.size * 100

    if coverage > 40 or coverage < 20:
        print(f"❌ {img_path.name}: mask coverage = {coverage:.1f}%")
    else:
        print(f"✅ {img_path.name}: mask coverage = {coverage:.1f}%")

    cv2.imshow(f"Check: {img_path.name}", blended)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Spot-check one label file
label = here("data/preprocessed/yolo/labels/train").glob("*.txt").__next__()
content = label.read_text().strip()

parts = content.split()
if parts[0] == "0":
    print(f"✅ Class: {parts[0]}")
else:
    print(f"❌ Class: {parts[0]}")

if len(parts) > 10:
    print(f"✅ Points: {(len(parts) - 1) // 2}")
else:
    print(f"❌ Points: {(len(parts) - 1) // 2}")

coords = [float(x) for x in parts[1:]]

if min(coords) >= 0 and max(coords) <= 1:
    print(f"✅ Coord range: {min(coords):.4f} – {max(coords):.4f}")
else:
    print(f"❌ Coord range: {min(coords):.4f} – {max(coords):.4f}")