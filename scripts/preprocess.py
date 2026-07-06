"""
HC18 Dataset Preprocessor
=========================

Reads HC18 training images and their paired *_Annotation.png binary masks,
then writes output in two model-ready formats:

  YOLOv8-seg  (segmentation)
  ─────────────────────────
  <output>/yolo/
      images/train/*.png     RGB ultrasound images
      images/val/*.png
      labels/train/*.txt     YOLO polygon labels  (class x1 y1 x2 y2 ...)
      labels/val/*.txt
      data.yaml              dataset config for ultralytics

  U-Net / FastAI  (semantic segmentation)
  ────────────────────────────────────────
  <output>/fastai/
      images/train/*.png     RGB ultrasound images
      images/val/*.png
      masks/train/*.png      binary masks  (0 = background, 255 = fetal head)
      masks/val/*.png
      split.csv              filename, split, pixel_size_mm, hc_mm

  <output>/pixel_metadata.json   per-image pixel_size_mm + hc_mm

Annotation convention
─────────────────────
HC18 ships a *_Annotation.png alongside every training image.
  - Shape : same (H, W) as the paired image
  - dtype : uint8
  - Values: 0 (background) | 255 (fetal head ellipse OUTLINE)

IMPORTANT: The annotation stores only the ellipse *outline* (hollow ring),
not a filled region. Raw coverage is ~0.4–0.9% of image area. The
fill_annotation_mask() function fills the interior so the mask becomes a
solid ellipse.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyprojroot import here
from sklearn.model_selection import train_test_split

DATASET_PATH = here("data/raw/training_set_pixel_size_and_HC.csv")
IMAGE_DIR = here("data/raw/training_set")
OUTPUT_DIR = here("data/preprocessed")


def fill_annotation_mask(outline: np.ndarray) -> np.ndarray:
    """
    Convert a hollow ellipse outline (as stored in HC18 *_Annotation.png
    files) into a solid filled binary mask.

    The HC18 annotations contain only the ellipse boundary — a thin ring
    of white pixels (~0.4–0.9 % image area). Models need a solid region.
    This function uses morphological closing to bridge small gaps, then
    fills the interior using the convex hull of the contour, yielding
    the expected 20–40 % mask coverage.

    Args:
        outline: Grayscale uint8 array (H, W) with values 0 or 255,
                 representing the ellipse boundary.

    Returns:
        Solid binary mask (H, W, uint8) with the ellipse interior filled
        (255 = fetal head, 0 = background). Returns a copy of ``outline``
        unchanged if no contour is found.
    """
    # Step 1: Morphological closing to bridge small gaps in the outline
    # This is crucial for annotations that have disconnected pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 2: Find contours in the closed mask
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return outline.copy()   # nothing to fill — return as-is

    # Step 3: Use the largest contour (guards against tiny noise blobs)
    largest = max(contours, key=cv2.contourArea)
    
    # Step 4: Compute convex hull to ensure a solid, filled region
    # This works better for ellipses than drawContours with FILLED
    hull = cv2.convexHull(largest)

    # Step 5: Fill the convex hull
    filled = np.zeros_like(outline)
    cv2.fillPoly(filled, [hull], 255)
    
    return filled


def mask_to_yolo_polygon(mask: np.ndarray) -> str | None:
    """
    Convert a binary mask (0/255, H×W uint8) to a YOLO segmentation label.

    YOLO segmentation format (one line per object):
        <class_id> <x1> <y1> <x2> <y2> ...   (all values normalised 0–1)

    Args:
        mask: Binary mask array, shape (H, W), values 0 or 255.

    Returns:
        Label string ready to write to a .txt file, or None if no contour
        was found (i.e. the mask is empty).
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Use the largest contour (the fetal head ellipse)
    contour = max(contours, key=cv2.contourArea)

    # Simplify polygon: epsilon = 0.5 % of perimeter keeps detail while
    # keeping point count manageable for YOLO training.
    epsilon = 0.005 * cv2.arcLength(contour, closed=True)
    contour = cv2.approxPolyDP(contour, epsilon, closed=True)

    if len(contour) < 3:
        return None  # degenerate contour — skip

    points = contour.reshape(-1, 2).astype(float)
    points[:, 0] /= w   # normalise x → [0, 1]
    points[:, 1] /= h   # normalise y → [0, 1]

    # Clamp to guard against floating-point edge cases
    points = np.clip(points, 0.0, 1.0)

    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"0 {coords}"   # class 0 = fetal_head


def preprocess(
    csv_path: Path,
    images_dir: Path,
    output_dir: Path,
    val_split: float = 0.15,
) -> None:
    """
    Run the full HC18 preprocessing pipeline.

    Args:
        csv_path:   Path to training_set_pixel_size_and_HC.csv
                    Expected columns: filename | pixel size(mm) | head circumference (mm)
        images_dir: Directory that contains both the raw PNGs and the
                    paired *_Annotation.png files.
        output_dir: Root output directory (will be created if absent).
        val_split:  Fraction of samples to reserve for validation.
    """
    # ------------------------------------------------------------------
    # Load & validate CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()   # strip accidental whitespace
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns : {df.columns.tolist()}")

    required = {"filename", "pixel size(mm)", "head circumference (mm)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ------------------------------------------------------------------
    # Pre-filter: Remove problematic samples BEFORE splitting
    # ------------------------------------------------------------------
    print(f"\nPre-filtering samples for quality...")
    valid_samples = []
    rejected_samples = []

    for _, row in df.iterrows():
        filename = str(row["filename"])
        stem = Path(filename).stem
        img_path = images_dir / filename
        ann_path = images_dir / f"{stem}_Annotation.png"

        # Check if files exist
        if not img_path.exists():
            rejected_samples.append((filename, "Missing image file"))
            continue
        if not ann_path.exists():
            rejected_samples.append((filename, "Missing annotation file"))
            continue

        # Load and validate annotation
        mask = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            rejected_samples.append((filename, "Failed to load annotation"))
            continue

        # Threshold and fill
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_filled = fill_annotation_mask(mask_binary)

        # Check if mask was successfully filled
        coverage = (mask_filled > 0).sum() / mask_filled.size * 100

        if coverage < 5.0:
            rejected_samples.append((filename, f"Hollow mask (coverage={coverage:.1f}%)"))
            continue
        if coverage > 60.0:
            rejected_samples.append((filename, f"Suspicious mask (coverage={coverage:.1f}%)"))
            continue

        # Validate that YOLO polygon can be generated
        test_polygon = mask_to_yolo_polygon(mask_filled)
        if test_polygon is None:
            rejected_samples.append((filename, "Failed to generate YOLO polygon"))
            continue

        # All checks passed
        valid_samples.append(row)

    # Create clean dataframe with only valid samples
    df_clean = pd.DataFrame(valid_samples)
    
    print(f"Pre-filtering results:")
    print(f"  Total samples:    {len(df)}")
    print(f"  Valid samples:    {len(df_clean)}")
    print(f"  Rejected samples: {len(rejected_samples)}")
    
    if rejected_samples:
        print(f"\nRejected samples:")
        for fname, reason in rejected_samples:
            print(f"  ✗ {fname:20s} — {reason}")

    if len(df_clean) == 0:
        raise ValueError("No valid samples remaining after pre-filtering!")

    # ------------------------------------------------------------------
    # Train / val split (on clean data only)
    # ------------------------------------------------------------------
    train_df, val_df = train_test_split(
        df_clean, test_size=val_split, random_state=67, shuffle=True
    )
    print(f"\nTrain/val split on clean data:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples:   {len(val_df)}")

    splits: dict[str, pd.DataFrame] = {"train": train_df, "val": val_df}

    # ------------------------------------------------------------------
    # Create output directory tree
    # ------------------------------------------------------------------
    for split in splits:
        # YOLOv8-seg
        (output_dir / "yolo" / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
        # U-Net / FastAI
        (output_dir / "fastai" / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "fastai" / "masks"  / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Process images (all samples have been pre-validated)
    # ------------------------------------------------------------------
    pixel_metadata: dict[str, dict[str, float]] = {}
    split_records:  list[dict] = []

    for split_name, split_df in splits.items():
        print(f"\nProcessing '{split_name}' split ({len(split_df)} images)…")

        for _, row in split_df.iterrows():
            filename = str(row["filename"])
            stem = Path(filename).stem  # e.g. "001_HC"
            img_path = images_dir / filename
            ann_path = images_dir / f"{stem}_Annotation.png"

            # ── load image ──────────────────────────────────────────────
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            # ── load & process annotation mask ─────────────────────────
            mask = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = fill_annotation_mask(mask)

            # ── metadata from CSV ───────────────────────────────────────
            pixel_size: float = float(row["pixel size(mm)"])
            hc_mm:      float = float(row["head circumference (mm)"])

            # ── YOLOv8-seg output ───────────────────────────────────────
            yolo_label = mask_to_yolo_polygon(mask)
            yolo_img_path = output_dir / "yolo" / "images" / split_name / f"{stem}.png"
            label_path = output_dir / "yolo" / "labels" / split_name / f"{stem}.txt"
            
            cv2.imwrite(str(yolo_img_path), img_rgb)
            label_path.write_text(yolo_label)

            # ── U-Net / FastAI output ───────────────────────────────────
            fastai_img_path  = output_dir / "fastai" / "images" / split_name / f"{stem}.png"
            fastai_mask_path = output_dir / "fastai" / "masks"  / split_name / f"{stem}.png"
            cv2.imwrite(str(fastai_img_path),  img_rgb)
            cv2.imwrite(str(fastai_mask_path), mask)

            # ── accumulate records ──────────────────────────────────────
            pixel_metadata[f"{stem}.png"] = {
                "pixel_size_mm": pixel_size,
                "hc_mm":         hc_mm,
            }
            split_records.append(
                {
                    "filename":      f"{stem}.png",
                    "split":         split_name,
                    "pixel_size_mm": pixel_size,
                    "hc_mm":         hc_mm,
                }
            )

    # ------------------------------------------------------------------
    # Write YOLOv8 data.yaml
    # ------------------------------------------------------------------
    data_yaml_path = output_dir / "yolo" / "data.yaml"
    data_yaml_path.write_text(
        f"path: {(output_dir / 'yolo').resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "nc: 1\n"
        "names: ['fetal_head']\n"
    )
    print(f"\ndata.yaml written  → {data_yaml_path}")

    # ------------------------------------------------------------------
    # Write FastAI split CSV
    # ------------------------------------------------------------------
    split_csv = output_dir / "fastai" / "split.csv"
    pd.DataFrame(split_records).to_csv(split_csv, index=False)
    print(f"split.csv written  → {split_csv}")

    # ------------------------------------------------------------------
    # Write pixel / HC metadata JSON
    # ------------------------------------------------------------------
    metadata_path = output_dir / "pixel_metadata.json"
    metadata_path.write_text(json.dumps(pixel_metadata, indent=2))
    print(f"metadata written   → {metadata_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    processed = len(split_records)
    print(f"\n{'='*60}")
    print("Preprocessing complete.")
    print(f"  Output root        : {output_dir}")
    print(f"  Original samples   : {len(df)}")
    print(f"  Rejected samples   : {len(rejected_samples)}")
    print(f"  Clean samples      : {len(df_clean)}")
    print(f"  Train samples      : {len(train_df)}")
    print(f"  Val samples        : {len(val_df)}")
    print(f"  Processed & written: {processed}")
    print(f"{'='*60}")
    
    if rejected_samples:
        print(f"\n💡 Tip: {len(rejected_samples)} samples were filtered out during pre-processing to ensure only high-quality data for the training. ")


if __name__ == "__main__":
    preprocess(
        csv_path=DATASET_PATH,
        images_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        val_split=0.15,
    )