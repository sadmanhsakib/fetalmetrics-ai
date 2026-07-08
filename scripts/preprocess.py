"""
preprocess.py
=============
HC18 Grand Challenge dataset preprocessor.

Reads the HC18 training images and their paired ``*_Annotation.png`` binary
masks, then writes output in two model-ready formats:

YOLOv8-seg (instance segmentation)
------------------------------------
``<output>/yolo/``
    images/train/*.png     RGB ultrasound images
    images/val/*.png
    labels/train/*.txt     YOLO polygon labels  (class x1 y1 x2 y2 …)
    labels/val/*.txt
    data.yaml              Ultralytics dataset configuration

U-Net / FastAI (semantic segmentation)
---------------------------------------
``<output>/fastai/``
    images/train/*.png     RGB ultrasound images
    images/val/*.png
    masks/train/*.png      Binary masks  (0 = background, 255 = fetal head)
    masks/val/*.png
    split.csv              filename | split | pixel_size_mm | hc_mm

``<output>/pixel_metadata.json``
    Per-image pixel_size_mm and hc_mm look-up table.

Annotation convention
----------------------
HC18 ships a ``*_Annotation.png`` alongside every training image.

* Shape : identical (H, W) to the paired image.
* Dtype : uint8.
* Values: 0 (background) | 255 (fetal-head ellipse **outline only**).

The annotation encodes only the ellipse *outline* (a hollow ring), so raw
foreground coverage is roughly 0.4–0.9 % of image area.  The
``fill_annotation_mask`` function fills the interior to produce a solid ellipse
region suitable for supervised segmentation training.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyprojroot import here
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Module-level path constants
# ---------------------------------------------------------------------------
DATASET_PATH = here("data/raw/training_set_pixel_size_and_HC.csv")
IMAGE_DIR = here("data/raw/training_set")
OUTPUT_DIR = here("data/preprocessed")


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def fill_annotation_mask(outline: np.ndarray) -> np.ndarray:
    """Convert a hollow ellipse outline into a solid filled binary mask.

    HC18 annotations store only the ellipse boundary — a thin ring of white
    pixels that represents roughly 0.4–0.9 % of image area.  Segmentation
    models require a solid foreground region.  This function bridges small
    gaps with morphological closing, identifies the largest contour, then
    fills its convex hull, yielding the expected 20–40 % foreground coverage.

    Parameters
    ----------
    outline:
        Grayscale uint8 array of shape (H, W) with pixel values 0 or 255
        representing the annotated ellipse boundary.

    Returns
    -------
    numpy.ndarray
        Solid binary mask of shape (H, W, uint8) where the ellipse interior
        is filled (255 = fetal head, 0 = background).  Returns a copy of
        ``outline`` unchanged if no contour is found.
    """
    # Morphological closing bridges disconnected outline pixels before
    # contour extraction.  Two iterations handle moderate annotation gaps.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No contour found — return the original outline unchanged.
        return outline.copy()

    # Guard against noise blobs by selecting only the largest contour.
    largest = max(contours, key=cv2.contourArea)

    # Fill the convex hull of the contour for robustness against concavities
    # that would leave interior gaps with a simple drawContours(FILLED) call.
    hull = cv2.convexHull(largest)
    filled = np.zeros_like(outline)
    cv2.fillPoly(filled, [hull], 255)

    return filled


def mask_to_yolo_polygon(mask: np.ndarray) -> str | None:
    """Convert a binary mask to a YOLO segmentation label string.

    YOLO segmentation format (one line per instance)::

        <class_id> <x1> <y1> <x2> <y2> …   (all coordinates normalised 0–1)

    The polygon is simplified with the Ramer–Douglas–Peucker algorithm
    (ε = 0.5 % of arc length) to reduce point count while preserving the
    ellipse boundary with sufficient fidelity.

    Parameters
    ----------
    mask:
        Binary mask array of shape (H, W), values 0 or 255.

    Returns
    -------
    str | None
        Label string ready to write to a ``.txt`` file, or ``None`` if no
        contour is found (i.e. the mask is empty or degenerate).
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    # Simplify the polygon boundary.  ε = 0.5 % of arc length balances
    # boundary fidelity against label file size.
    epsilon = 0.005 * cv2.arcLength(contour, closed=True)
    contour = cv2.approxPolyDP(contour, epsilon, closed=True)

    if len(contour) < 3:
        # Degenerate contour — cannot form a valid polygon.
        return None

    points = contour.reshape(-1, 2).astype(float)
    points[:, 0] /= w  # Normalise x to [0, 1].
    points[:, 1] /= h  # Normalise y to [0, 1].

    # Guard against floating-point rounding outside [0, 1].
    points = np.clip(points, 0.0, 1.0)

    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"0 {coords}"  # Class 0 = fetal_head.


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def preprocess(
    csv_path: Path,
    images_dir: Path,
    output_dir: Path,
    val_split: float = 0.15,
) -> None:
    """Execute the full HC18 preprocessing pipeline.

    Parameters
    ----------
    csv_path:
        Path to ``training_set_pixel_size_and_HC.csv``.  Expected columns:
        ``filename``, ``pixel size(mm)``, ``head circumference (mm)``.
    images_dir:
        Directory containing raw PNG scans and paired ``*_Annotation.png``
        files.
    output_dir:
        Root output directory.  Created automatically if absent.
    val_split:
        Fraction of quality-filtered samples reserved for validation
        (default 0.15 = 15 %).

    Raises
    ------
    ValueError
        If required CSV columns are absent, or if no samples survive the
        quality pre-filter.
    """
    # -----------------------------------------------------------------------
    # Load and validate the metadata CSV
    # -----------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Strip accidental leading/trailing whitespace.
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns : {df.columns.tolist()}")

    required = {"filename", "pixel size(mm)", "head circumference (mm)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # -----------------------------------------------------------------------
    # Quality pre-filter: discard samples before the train/val split so that
    # the split ratio is computed only over clean, usable examples.
    # -----------------------------------------------------------------------
    print("\nPre-filtering samples for quality…")
    valid_samples: list = []
    rejected_samples: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        filename = str(row["filename"])
        stem = Path(filename).stem
        img_path = images_dir / filename
        ann_path = images_dir / f"{stem}_Annotation.png"

        # Verify both the image and its paired annotation exist on disk.
        if not img_path.exists():
            rejected_samples.append((filename, "Missing image file"))
            continue
        if not ann_path.exists():
            rejected_samples.append((filename, "Missing annotation file"))
            continue

        mask = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            rejected_samples.append((filename, "Failed to load annotation"))
            continue

        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_filled = fill_annotation_mask(mask_binary)

        # Reject masks where the fill algorithm failed (hollow outline) or
        # where coverage is implausibly large (likely a corrupted annotation).
        coverage = (mask_filled > 0).sum() / mask_filled.size * 100
        if coverage < 5.0:
            rejected_samples.append((filename, f"Hollow mask (coverage={coverage:.1f}%)"))
            continue
        if coverage > 60.0:
            rejected_samples.append((filename, f"Suspicious mask (coverage={coverage:.1f}%)"))
            continue

        # Ensure a valid YOLO polygon can be generated from the filled mask.
        if mask_to_yolo_polygon(mask_filled) is None:
            rejected_samples.append((filename, "Failed to generate YOLO polygon"))
            continue

        valid_samples.append(row)

    df_clean = pd.DataFrame(valid_samples)

    print(
        f"Pre-filtering results:\n"
        f"  Total samples    : {len(df)}\n"
        f"  Valid samples    : {len(df_clean)}\n"
        f"  Rejected samples : {len(rejected_samples)}"
    )
    if rejected_samples:
        print("\nRejected samples:")
        for fname, reason in rejected_samples:
            print(f"  ✗ {fname:<20s} — {reason}")

    if len(df_clean) == 0:
        raise ValueError("No valid samples remaining after pre-filtering.")

    # -----------------------------------------------------------------------
    # Train / validation split (applied only to the quality-filtered set)
    # -----------------------------------------------------------------------
    train_df, val_df = train_test_split(
        df_clean, test_size=val_split, random_state=67, shuffle=True
    )
    print(
        f"\nTrain/val split:\n"
        f"  Train: {len(train_df)}\n"
        f"  Val  : {len(val_df)}"
    )
    splits: dict[str, pd.DataFrame] = {"train": train_df, "val": val_df}

    # -----------------------------------------------------------------------
    # Output directory tree creation
    # -----------------------------------------------------------------------
    for split in splits:
        (output_dir / "yolo" / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "fastai" / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "fastai" / "masks" / split).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Image processing loop
    # All samples have been pre-validated, so no per-sample error handling is
    # needed beyond guard-rails already embedded in the helper functions.
    # -----------------------------------------------------------------------
    pixel_metadata: dict[str, dict[str, float]] = {}
    split_records: list[dict] = []

    for split_name, split_df in splits.items():
        print(f"\nProcessing '{split_name}' split ({len(split_df)} images)…")

        for _, row in split_df.iterrows():
            filename = str(row["filename"])
            stem = Path(filename).stem
            img_path = images_dir / filename
            ann_path = images_dir / f"{stem}_Annotation.png"

            # Load as grayscale and convert to 3-channel RGB so both model
            # families receive a consistent (H, W, 3) uint8 input format.
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            # Load, threshold, and fill the annotation mask.
            mask = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = fill_annotation_mask(mask)

            pixel_size: float = float(row["pixel size(mm)"])
            hc_mm: float = float(row["head circumference (mm)"])

            # -- YOLOv8-seg output ------------------------------------------
            yolo_label = mask_to_yolo_polygon(mask)
            yolo_img_path = (
                output_dir / "yolo" / "images" / split_name / f"{stem}.png"
            )
            label_path = output_dir / "yolo" / "labels" / split_name / f"{stem}.txt"

            cv2.imwrite(str(yolo_img_path), img_rgb)
            label_path.write_text(yolo_label)

            # -- U-Net / FastAI output --------------------------------------
            fastai_img_path = (
                output_dir / "fastai" / "images" / split_name / f"{stem}.png"
            )
            fastai_mask_path = (
                output_dir / "fastai" / "masks" / split_name / f"{stem}.png"
            )
            cv2.imwrite(str(fastai_img_path), img_rgb)
            cv2.imwrite(str(fastai_mask_path), mask)

            # -- Accumulate metadata records --------------------------------
            pixel_metadata[f"{stem}.png"] = {
                "pixel_size_mm": pixel_size,
                "hc_mm": hc_mm,
            }
            split_records.append(
                {
                    "filename": f"{stem}.png",
                    "split": split_name,
                    "pixel_size_mm": pixel_size,
                    "hc_mm": hc_mm,
                }
            )

    # -----------------------------------------------------------------------
    # Write output artefacts
    # -----------------------------------------------------------------------

    # Ultralytics data.yaml — required by ``yolo train``.
    data_yaml_path = output_dir / "yolo" / "data.yaml"
    data_yaml_path.write_text(
        f"path: {(output_dir / 'yolo').resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "nc: 1\n"
        "names: ['fetal_head']\n"
    )
    print(f"\ndata.yaml written  → {data_yaml_path}")

    # FastAI split index CSV.
    split_csv = output_dir / "fastai" / "split.csv"
    pd.DataFrame(split_records).to_csv(split_csv, index=False)
    print(f"split.csv written  → {split_csv}")

    # Per-image pixel-size and HC look-up JSON.
    metadata_path = output_dir / "pixel_metadata.json"
    metadata_path.write_text(json.dumps(pixel_metadata, indent=2))
    print(f"metadata written   → {metadata_path}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    processed = len(split_records)
    print(f"\n{'=' * 60}")
    print("Preprocessing complete.")
    print(f"  Output root        : {output_dir}")
    print(f"  Original samples   : {len(df)}")
    print(f"  Rejected samples   : {len(rejected_samples)}")
    print(f"  Clean samples      : {len(df_clean)}")
    print(f"  Train samples      : {len(train_df)}")
    print(f"  Val samples        : {len(val_df)}")
    print(f"  Written samples    : {processed}")
    print(f"{'=' * 60}")

    if rejected_samples:
        print(
            f"\nNote: {len(rejected_samples)} samples were excluded during "
            "pre-filtering to ensure only high-quality pairs enter training."
        )


if __name__ == "__main__":
    preprocess(
        csv_path=DATASET_PATH,
        images_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        val_split=0.15,
    )