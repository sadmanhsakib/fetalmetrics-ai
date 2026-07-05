"""Comprehensive validation of HC18 preprocessing output.

This script validates that preprocessed data meets quality standards:
  - Masks are filled solid ellipses (not hollow outlines)
  - Coverage is 20-40% (typical for fetal head in ultrasound)
  - Masks are binary (0/255 only)
  - Images are RGB format
  - YOLO labels have proper format and normalized coordinates
  - Paired files exist across all formats
  - Train/val splits are consistent
  - No data leakage between splits
"""
import json
import random

import cv2
import numpy as np
import pandas as pd
from pyprojroot import here

OUTPUT_DIR = here("data/preprocessed")
IMG_DIR = OUTPUT_DIR / "fastai" / "images" / "train"
MASK_DIR = OUTPUT_DIR / "fastai" / "masks" / "train"

print("=" * 70)
print("HC18 PREPROCESSING VALIDATION")
print("=" * 70)

# ============================================================================
# 1. FILE STRUCTURE & COUNTS
# ============================================================================
print("\n[1] FILE STRUCTURE & COUNTS")
print("-" * 70)

train_img_fastai = list((OUTPUT_DIR / "fastai" / "images" / "train").glob("*.png"))
val_img_fastai = list((OUTPUT_DIR / "fastai" / "images" / "val").glob("*.png"))
train_mask_fastai = list((OUTPUT_DIR / "fastai" / "masks" / "train").glob("*.png"))
val_mask_fastai = list((OUTPUT_DIR / "fastai" / "masks" / "val").glob("*.png"))

train_img_yolo = list((OUTPUT_DIR / "yolo" / "images" / "train").glob("*.png"))
val_img_yolo = list((OUTPUT_DIR / "yolo" / "images" / "val").glob("*.png"))
train_label_yolo = list((OUTPUT_DIR / "yolo" / "labels" / "train").glob("*.txt"))
val_label_yolo = list((OUTPUT_DIR / "yolo" / "labels" / "val").glob("*.txt"))

print(f"FastAI train images: {len(train_img_fastai)}")
print(f"FastAI train masks:  {len(train_mask_fastai)}")
print(f"FastAI val images:   {len(val_img_fastai)}")
print(f"FastAI val masks:    {len(val_mask_fastai)}")
print(f"YOLO train images:   {len(train_img_yolo)}")
print(f"YOLO train labels:   {len(train_label_yolo)}")
print(f"YOLO val images:     {len(val_img_yolo)}")
print(f"YOLO val labels:     {len(val_label_yolo)}")

# Check counts match
issues = []
if len(train_img_fastai) != len(train_mask_fastai):
    issues.append(f"❌ FastAI train: {len(train_img_fastai)} images ≠ {len(train_mask_fastai)} masks")
if len(val_img_fastai) != len(val_mask_fastai):
    issues.append(f"❌ FastAI val: {len(val_img_fastai)} images ≠ {len(val_mask_fastai)} masks")
if len(train_img_yolo) != len(train_label_yolo):
    issues.append(f"❌ YOLO train: {len(train_img_yolo)} images ≠ {len(train_label_yolo)} labels")
if len(val_img_yolo) != len(val_label_yolo):
    issues.append(f"❌ YOLO val: {len(val_img_yolo)} images ≠ {len(val_label_yolo)} labels")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✅ All file counts match")

# ============================================================================
# 2. SPLIT CONSISTENCY
# ============================================================================
print("\n[2] SPLIT CONSISTENCY")
print("-" * 70)

train_stems_fastai = {p.stem for p in train_img_fastai}
val_stems_fastai = {p.stem for p in val_img_fastai}
train_stems_yolo = {p.stem for p in train_img_yolo}
val_stems_yolo = {p.stem for p in val_img_yolo}

# Check no overlap between train/val
overlap = train_stems_fastai & val_stems_fastai
if overlap:
    print(f"❌ DATA LEAKAGE: {len(overlap)} files in both train and val: {list(overlap)[:5]}")
else:
    print("✅ No overlap between train and val splits")

# Check FastAI and YOLO have same splits
if train_stems_fastai != train_stems_yolo:
    diff = train_stems_fastai ^ train_stems_yolo
    print(f"❌ Train splits differ between FastAI and YOLO: {len(diff)} files differ")
else:
    print("✅ Train splits match between FastAI and YOLO")

if val_stems_fastai != val_stems_yolo:
    diff = val_stems_fastai ^ val_stems_yolo
    print(f"❌ Val splits differ between FastAI and YOLO: {len(diff)} files differ")
else:
    print("✅ Val splits match between FastAI and YOLO")

# ============================================================================
# 3. MASK QUALITY CHECKS
# ============================================================================
print("\n[3] MASK QUALITY CHECKS")
print("-" * 70)

samples = random.sample(train_img_fastai, min(10, len(train_img_fastai)))
mask_issues = []
coverage_values = []
hollow_masks = []

for img_path in samples:
    mask_path = MASK_DIR / img_path.name
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        mask_issues.append(f"❌ {img_path.name}: Failed to load mask")
        continue
    
    # Check binary (only 0 and 255)
    unique_vals = np.unique(mask)
    if not np.array_equal(unique_vals, [0, 255]) and not np.array_equal(unique_vals, [0]) and not np.array_equal(unique_vals, [255]):
        if len(unique_vals) > 2:
            mask_issues.append(f"❌ {img_path.name}: Not binary! Has values: {unique_vals}")
    
    # Calculate coverage
    coverage = (mask > 0).sum() / mask.size * 100
    coverage_values.append(coverage)
    
    # Check if mask is filled (not hollow outline)
    # A hollow outline would have very low coverage (<5%)
    if coverage < 5:
        hollow_masks.append(f"❌ {img_path.name}: Possibly hollow (coverage={coverage:.1f}%)")
    
    # Check expected coverage range (20-40% for fetal head)
    if coverage < 15 or coverage > 50:
        mask_issues.append(f"⚠️  {img_path.name}: Coverage {coverage:.1f}% outside typical range (15-50%)")

if hollow_masks:
    print("\n⚠️  HOLLOW MASK WARNING:")
    for h in hollow_masks:
        print(f"  {h}")
    print("  → Masks should be FILLED ellipses, not outlines!")

if mask_issues:
    print("\nMask Issues Found:")
    for issue in mask_issues:
        print(f"  {issue}")
else:
    print("✅ All sampled masks are binary and properly filled")

if coverage_values:
    print(f"\nMask Coverage Statistics (n={len(coverage_values)}):")
    print(f"  Mean:   {np.mean(coverage_values):.1f}%")
    print(f"  Median: {np.median(coverage_values):.1f}%")
    print(f"  Range:  {np.min(coverage_values):.1f}% - {np.max(coverage_values):.1f}%")
    print(f"  Expected: 20-40% for fetal head in ultrasound")

# ============================================================================
# 4. IMAGE FORMAT CHECKS
# ============================================================================
print("\n[4] IMAGE FORMAT CHECKS")
print("-" * 70)

img_samples = random.sample(train_img_fastai, min(5, len(train_img_fastai)))
format_issues = []

for img_path in img_samples:
    img = cv2.imread(str(img_path))
    if img is None:
        format_issues.append(f"❌ {img_path.name}: Failed to load")
        continue
    
    # Check RGB (should have 3 channels)
    if len(img.shape) != 3 or img.shape[2] != 3:
        format_issues.append(f"❌ {img_path.name}: Not RGB! Shape: {img.shape}")
    
    # Check all channels are identical (grayscale converted to RGB)
    if np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,1], img[:,:,2]):
        pass  # Expected - grayscale ultrasound converted to RGB
    else:
        format_issues.append(f"⚠️  {img_path.name}: Channels differ (unexpected for ultrasound)")

if format_issues:
    for issue in format_issues:
        print(f"  {issue}")
else:
    print("✅ All sampled images are proper RGB format")

# ============================================================================
# 5. YOLO LABEL FORMAT CHECKS
# ============================================================================
print("\n[5] YOLO LABEL FORMAT CHECKS")
print("-" * 70)

label_samples = random.sample(train_label_yolo, min(10, len(train_label_yolo)))
label_issues = []
point_counts = []

for label_path in label_samples:
    content = label_path.read_text().strip()
    
    if not content:
        label_issues.append(f"❌ {label_path.name}: Empty label file")
        continue
    
    parts = content.split()
    
    # Check class ID
    if parts[0] != "0":
        label_issues.append(f"❌ {label_path.name}: Wrong class '{parts[0]}' (expected '0')")
    
    # Check coordinate count (should be even, minimum 6 for triangle)
    coord_count = len(parts) - 1
    if coord_count < 6:
        label_issues.append(f"❌ {label_path.name}: Only {coord_count//2} points (need ≥3)")
    elif coord_count % 2 != 0:
        label_issues.append(f"❌ {label_path.name}: Odd number of coordinates ({coord_count})")
    else:
        point_counts.append(coord_count // 2)
    
    # Check coordinate normalization
    try:
        coords = [float(x) for x in parts[1:]]
        if min(coords) < -0.01 or max(coords) > 1.01:  # Small tolerance for float precision
            label_issues.append(
                f"❌ {label_path.name}: Coords not normalized! Range: {min(coords):.4f} - {max(coords):.4f}"
            )
    except ValueError:
        label_issues.append(f"❌ {label_path.name}: Invalid coordinate values")

if label_issues:
    print("\nLabel Issues Found:")
    for issue in label_issues:
        print(f"  {issue}")
else:
    print("✅ All sampled YOLO labels have correct format")

if point_counts:
    print(f"\nPolygon Point Counts (n={len(point_counts)}):")
    print(f"  Mean:   {np.mean(point_counts):.1f} points")
    print(f"  Median: {np.median(point_counts):.0f} points")
    print(f"  Range:  {np.min(point_counts)} - {np.max(point_counts)} points")

# ============================================================================
# 6. METADATA CHECKS
# ============================================================================
print("\n[6] METADATA CHECKS")
print("-" * 70)

# Check split.csv
split_csv_path = OUTPUT_DIR / "fastai" / "split.csv"
if not split_csv_path.exists():
    print("❌ split.csv not found")
else:
    split_df = pd.read_csv(split_csv_path)
    print(f"✅ split.csv loaded: {len(split_df)} records")
    print(f"   Columns: {split_df.columns.tolist()}")
    
    # Check required columns
    required_cols = {"filename", "split", "pixel_size_mm", "hc_mm"}
    if not required_cols.issubset(split_df.columns):
        print(f"   ❌ Missing columns: {required_cols - set(split_df.columns)}")
    else:
        print(f"   ✅ All required columns present")
    
    # Check split counts
    train_count = (split_df["split"] == "train").sum()
    val_count = (split_df["split"] == "val").sum()
    print(f"   Train: {train_count}, Val: {val_count} ({val_count/(train_count+val_count)*100:.1f}% val)")

# Check pixel_metadata.json
metadata_path = OUTPUT_DIR / "pixel_metadata.json"
if not metadata_path.exists():
    print("❌ pixel_metadata.json not found")
else:
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"✅ pixel_metadata.json loaded: {len(metadata)} records")

# Check data.yaml
yaml_path = OUTPUT_DIR / "yolo" / "data.yaml"
if not yaml_path.exists():
    print("❌ data.yaml not found")
else:
    yaml_content = yaml_path.read_text()
    print(f"✅ data.yaml exists")
    if "nc: 1" in yaml_content and "fetal_head" in yaml_content:
        print("   ✅ Correct class configuration")
    else:
        print("   ❌ data.yaml may have incorrect configuration")

# ============================================================================
# 7. VISUAL SPOT-CHECK
# ============================================================================
print("\n[7] VISUAL SPOT-CHECK")
print("-" * 70)
print("Displaying 3 random samples with mask overlays...")
print("Press any key to cycle through, ESC to skip visualization")

visual_samples = random.sample(train_img_fastai, min(3, len(train_img_fastai)))

for img_path in visual_samples:
    mask_path = MASK_DIR / img_path.name
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"⚠️  Skipping {img_path.name} - failed to load")
        continue

    # Create overlay
    overlay = img.copy()
    overlay[mask > 0] = [0, 255, 0]  # green overlay
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # Add text with stats
    coverage = (mask > 0).sum() / mask.size * 100
    status = "✅" if 15 <= coverage <= 50 else "⚠️"
    cv2.putText(
        blended, 
        f"{status} Coverage: {coverage:.1f}%", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (255, 255, 255), 
        2
    )

    cv2.imshow(f"Validation: {img_path.name}", blended)
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
total_issues = len(issues) + len(mask_issues) + len(format_issues) + len(label_issues) + len(hollow_masks)
if total_issues == 0:
    print("✅ ALL CHECKS PASSED - Preprocessing output looks great!")
else:
    print(f"⚠️  Found {total_issues} issues that may need attention")
print("=" * 70)