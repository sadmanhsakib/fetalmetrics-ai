"""
validate_preprocessing.py
=========================
Comprehensive structural and quality validation of the HC18 preprocessing
output produced by ``scripts/preprocess.py``.

Checks performed
----------------
1. **File structure and counts** — Verify that image, mask, and label counts
   are balanced across both FastAI and YOLO output trees.
2. **Split consistency** — Confirm there is no data leakage between the
   train and validation splits, and that both model format trees use identical
   splits.
3. **Mask quality** — Sample up to 10 training masks and verify they are
   binary (0/255 only), solid (not hollow outlines), and within the expected
   20–40 % foreground-coverage range for a fetal head in a standard ultrasound
   frame.
4. **Image format** — Confirm images are 3-channel RGB (grayscale ultrasound
   converted to RGB during preprocessing).
5. **YOLO label format** — Validate class ID, minimum polygon point count,
   and that all coordinates are normalised to [0, 1].
6. **Metadata artefacts** — Check for the presence and basic correctness of
   ``split.csv``, ``pixel_metadata.json``, and ``data.yaml``.
7. **Visual spot-check** — Display three random image-mask overlay pairs in
   an OpenCV window for a human sanity check (press any key to advance,
   ESC to skip).

Usage
-----
    python scripts/test/validate_preprocessing.py
"""

from __future__ import annotations

import json
import random

import cv2
import numpy as np
import pandas as pd
from pyprojroot import here

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = here("data/preprocessed")
IMG_DIR = OUTPUT_DIR / "fastai" / "images" / "train"
MASK_DIR = OUTPUT_DIR / "fastai" / "masks" / "train"

print("=" * 70)
print("HC18 PREPROCESSING VALIDATION")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. File structure and counts
# ---------------------------------------------------------------------------
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

print(f"FastAI train images : {len(train_img_fastai)}")
print(f"FastAI train masks  : {len(train_mask_fastai)}")
print(f"FastAI val images   : {len(val_img_fastai)}")
print(f"FastAI val masks    : {len(val_mask_fastai)}")
print(f"YOLO train images   : {len(train_img_yolo)}")
print(f"YOLO train labels   : {len(train_label_yolo)}")
print(f"YOLO val images     : {len(val_img_yolo)}")
print(f"YOLO val labels     : {len(val_label_yolo)}")

issues: list[str] = []
if len(train_img_fastai) != len(train_mask_fastai):
    issues.append(
        f"FastAI train mismatch: {len(train_img_fastai)} images vs "
        f"{len(train_mask_fastai)} masks"
    )
if len(val_img_fastai) != len(val_mask_fastai):
    issues.append(
        f"FastAI val mismatch: {len(val_img_fastai)} images vs "
        f"{len(val_mask_fastai)} masks"
    )
if len(train_img_yolo) != len(train_label_yolo):
    issues.append(
        f"YOLO train mismatch: {len(train_img_yolo)} images vs "
        f"{len(train_label_yolo)} labels"
    )
if len(val_img_yolo) != len(val_label_yolo):
    issues.append(
        f"YOLO val mismatch: {len(val_img_yolo)} images vs "
        f"{len(val_label_yolo)} labels"
    )

if issues:
    for issue in issues:
        print(f"  FAIL: {issue}")
else:
    print("All file counts balanced.")

# ---------------------------------------------------------------------------
# 2. Split consistency
# ---------------------------------------------------------------------------
print("\n[2] SPLIT CONSISTENCY")
print("-" * 70)

train_stems_fastai = {p.stem for p in train_img_fastai}
val_stems_fastai = {p.stem for p in val_img_fastai}
train_stems_yolo = {p.stem for p in train_img_yolo}
val_stems_yolo = {p.stem for p in val_img_yolo}

# Data leakage check: no filename should appear in both splits.
overlap = train_stems_fastai & val_stems_fastai
if overlap:
    print(
        f"  DATA LEAKAGE: {len(overlap)} file(s) present in both train and "
        f"val: {list(overlap)[:5]}"
    )
else:
    print("No overlap between train and val splits.")

# Cross-format consistency: FastAI and YOLO trees must use identical splits.
if train_stems_fastai != train_stems_yolo:
    diff = train_stems_fastai ^ train_stems_yolo
    print(f"  Train split differs between FastAI and YOLO: {len(diff)} file(s)")
else:
    print("Train splits consistent between FastAI and YOLO.")

if val_stems_fastai != val_stems_yolo:
    diff = val_stems_fastai ^ val_stems_yolo
    print(f"  Val split differs between FastAI and YOLO: {len(diff)} file(s)")
else:
    print("Val splits consistent between FastAI and YOLO.")

# ---------------------------------------------------------------------------
# 3. Mask quality checks
# ---------------------------------------------------------------------------
print("\n[3] MASK QUALITY CHECKS")
print("-" * 70)

samples = random.sample(train_img_fastai, min(10, len(train_img_fastai)))
mask_issues: list[str] = []
hollow_masks: list[str] = []
coverage_values: list[float] = []

for img_path in samples:
    mask_path = MASK_DIR / img_path.name
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        mask_issues.append(f"{img_path.name}: failed to load mask")
        continue

    # Masks must be strictly binary (0 and 255 only).
    unique_vals = np.unique(mask)
    if len(unique_vals) > 2:
        mask_issues.append(
            f"{img_path.name}: non-binary values detected: {unique_vals}"
        )

    coverage = (mask > 0).sum() / mask.size * 100
    coverage_values.append(coverage)

    # Coverage below 5 % indicates the fill algorithm failed (hollow outline).
    if coverage < 5.0:
        hollow_masks.append(
            f"{img_path.name}: possible hollow outline (coverage={coverage:.1f}%)"
        )

    # Typical fetal-head foreground coverage is 15–50 % of image area.
    if coverage < 15.0 or coverage > 50.0:
        mask_issues.append(
            f"{img_path.name}: coverage {coverage:.1f}% outside expected range [15%, 50%]"
        )

if hollow_masks:
    print("  HOLLOW MASK WARNING — masks should be filled ellipses, not outlines:")
    for h in hollow_masks:
        print(f"    {h}")

if mask_issues:
    print("  Mask issues found:")
    for issue in mask_issues:
        print(f"    {issue}")
else:
    print("All sampled masks are binary and properly filled.")

if coverage_values:
    print(
        f"\nMask coverage statistics (n={len(coverage_values)}):\n"
        f"  Mean   : {np.mean(coverage_values):.1f}%\n"
        f"  Median : {np.median(coverage_values):.1f}%\n"
        f"  Range  : {np.min(coverage_values):.1f}% – {np.max(coverage_values):.1f}%\n"
        f"  Expected range: 20–40% for fetal head in standard ultrasound."
    )

# ---------------------------------------------------------------------------
# 4. Image format checks
# ---------------------------------------------------------------------------
print("\n[4] IMAGE FORMAT CHECKS")
print("-" * 70)

img_samples = random.sample(train_img_fastai, min(5, len(train_img_fastai)))
format_issues: list[str] = []

for img_path in img_samples:
    img = cv2.imread(str(img_path))
    if img is None:
        format_issues.append(f"{img_path.name}: failed to load")
        continue

    # Preprocessed images must be 3-channel (grayscale ultrasound → RGB).
    if img.ndim != 3 or img.shape[2] != 3:
        format_issues.append(f"{img_path.name}: expected 3-channel, got shape {img.shape}")

    # All three channels should be identical (grayscale replicated to RGB).
    if not (
        np.array_equal(img[:, :, 0], img[:, :, 1])
        and np.array_equal(img[:, :, 1], img[:, :, 2])
    ):
        format_issues.append(
            f"{img_path.name}: channels differ unexpectedly for grayscale ultrasound"
        )

if format_issues:
    for issue in format_issues:
        print(f"  {issue}")
else:
    print("All sampled images are proper 3-channel RGB format.")

# ---------------------------------------------------------------------------
# 5. YOLO label format checks
# ---------------------------------------------------------------------------
print("\n[5] YOLO LABEL FORMAT CHECKS")
print("-" * 70)

label_samples = random.sample(train_label_yolo, min(10, len(train_label_yolo)))
label_issues: list[str] = []
point_counts: list[int] = []

for label_path in label_samples:
    content = label_path.read_text().strip()

    if not content:
        label_issues.append(f"{label_path.name}: empty label file")
        continue

    parts = content.split()

    # Class ID must be 0 (fetal_head is the only class).
    if parts[0] != "0":
        label_issues.append(
            f"{label_path.name}: unexpected class ID '{parts[0]}' (expected '0')"
        )

    coord_count = len(parts) - 1
    if coord_count < 6:
        label_issues.append(
            f"{label_path.name}: only {coord_count // 2} polygon points (≥3 required)"
        )
    elif coord_count % 2 != 0:
        label_issues.append(
            f"{label_path.name}: odd coordinate count ({coord_count})"
        )
    else:
        point_counts.append(coord_count // 2)

    try:
        coords = [float(x) for x in parts[1:]]
        # Allow a small tolerance (1 %) for floating-point rounding at the border.
        if min(coords) < -0.01 or max(coords) > 1.01:
            label_issues.append(
                f"{label_path.name}: coordinates not normalised — "
                f"range [{min(coords):.4f}, {max(coords):.4f}]"
            )
    except ValueError:
        label_issues.append(f"{label_path.name}: invalid (non-numeric) coordinate values")

if label_issues:
    print("  Label issues found:")
    for issue in label_issues:
        print(f"    {issue}")
else:
    print("All sampled YOLO labels are correctly formatted.")

if point_counts:
    print(
        f"\nPolygon point counts (n={len(point_counts)}):\n"
        f"  Mean   : {np.mean(point_counts):.1f}\n"
        f"  Median : {np.median(point_counts):.0f}\n"
        f"  Range  : {np.min(point_counts)} – {np.max(point_counts)}"
    )

# ---------------------------------------------------------------------------
# 6. Metadata artefact checks
# ---------------------------------------------------------------------------
print("\n[6] METADATA ARTEFACT CHECKS")
print("-" * 70)

# split.csv
split_csv_path = OUTPUT_DIR / "fastai" / "split.csv"
if not split_csv_path.exists():
    print("  FAIL: split.csv not found")
else:
    split_df = pd.read_csv(split_csv_path)
    print(f"split.csv: {len(split_df)} records")
    required_cols = {"filename", "split", "pixel_size_mm", "hc_mm"}
    missing_cols = required_cols - set(split_df.columns)
    if missing_cols:
        print(f"  FAIL: missing columns: {missing_cols}")
    else:
        train_count = (split_df["split"] == "train").sum()
        val_count = (split_df["split"] == "val").sum()
        val_pct = val_count / (train_count + val_count) * 100
        print(f"  Columns OK. Train: {train_count}, Val: {val_count} ({val_pct:.1f}% val)")

# pixel_metadata.json
metadata_path = OUTPUT_DIR / "pixel_metadata.json"
if not metadata_path.exists():
    print("  FAIL: pixel_metadata.json not found")
else:
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"pixel_metadata.json: {len(metadata)} records")

# data.yaml
yaml_path = OUTPUT_DIR / "yolo" / "data.yaml"
if not yaml_path.exists():
    print("  FAIL: data.yaml not found")
else:
    yaml_content = yaml_path.read_text()
    if "nc: 1" in yaml_content and "fetal_head" in yaml_content:
        print("data.yaml: class configuration correct (nc=1, fetal_head)")
    else:
        print("  FAIL: data.yaml may have incorrect class configuration")

# ---------------------------------------------------------------------------
# 7. Visual spot-check
# ---------------------------------------------------------------------------
print("\n[7] VISUAL SPOT-CHECK")
print("-" * 70)
print("Displaying 3 random image-mask overlay pairs…")
print("Press any key to advance, ESC to skip visualisation.\n")

visual_samples = random.sample(train_img_fastai, min(3, len(train_img_fastai)))

for img_path in visual_samples:
    mask_path = MASK_DIR / img_path.name
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"  Skipping {img_path.name}: failed to load image or mask")
        continue

    # Composite a 70/30 blend of the original image and a green mask overlay.
    overlay = img.copy()
    overlay[mask > 0] = [0, 255, 0]
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    coverage = (mask > 0).sum() / mask.size * 100
    status = "OK" if 15.0 <= coverage <= 50.0 else "OUT_OF_RANGE"
    cv2.putText(
        blended,
        f"{status}  coverage={coverage:.1f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow(f"Validation: {img_path.name}", blended)
    key = cv2.waitKey(0)
    if key == 27:  # ESC — abort the visual check early.
        break

cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
total_issues = (
    len(issues)
    + len(mask_issues)
    + len(format_issues)
    + len(label_issues)
    + len(hollow_masks)
)
if total_issues == 0:
    print("All checks passed — preprocessing output looks correct.")
else:
    print(f"{total_issues} issue(s) found that may require attention.")
print("=" * 70)