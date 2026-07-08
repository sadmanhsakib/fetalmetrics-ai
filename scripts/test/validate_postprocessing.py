"""
validate_postprocessing.py
==========================
End-to-end validation of the post-processing pipeline and head-circumference
measurement accuracy for a selected segmentation model.

Evaluation protocol
--------------------
1. Prompt the operator to choose a segmentation model (``unet`` or
   ``yolov8``).
2. Load the HC18 ground-truth CSV to obtain reference pixel sizes and
   expected head-circumference (HC) values.
3. Sample up to 10 validation images, run full inference, apply the
   ``measure_hc`` geometric pipeline (mask cleaning, contour extraction,
   ellipse fitting, Ramanujan perimeter), and compute the absolute HC error
   in millimetres against the HC18 ground truth.
4. Report per-image results and aggregate Mean Absolute Error (MAE) and
   standard deviation.

Usage
-----
    python scripts/test/validate_postprocessing.py
"""

from __future__ import annotations

import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
from pyprojroot import here

# Ensure the src package is importable when running this script directly.
sys.path.insert(0, str(here("src")))

from config import MODELS
from inference.unet_onnx import UNetSegmenter
from inference.yolov8_onnx import YOLOv8Segmenter
from postprocess.ellipse import measure_hc

# Path to the HC18 calibration and ground-truth CSV.
_HC18_CSV = here("data/raw/training_set_pixel_size_and_HC.csv")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_image_rgb(img_path: os.PathLike) -> np.ndarray:
    """Load an image from disk as an RGB uint8 array.

    Both ``UNetSegmenter`` and ``YOLOv8Segmenter`` accept raw RGB images and
    handle all model-specific preprocessing (resize, normalization,
    letterboxing) internally inside their ``predict`` method.

    Parameters
    ----------
    img_path:
        Path to the input image file.

    Returns
    -------
    numpy.ndarray
        RGB uint8 array of shape (H, W, 3).
    """
    img = cv2.imread(str(img_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_hc18_data() -> dict[str, tuple[float, float]] | None:
    """Parse the HC18 CSV into a filename-keyed look-up table.

    Returns
    -------
    dict[str, tuple[float, float]] | None
        Mapping from image filename to ``(pixel_size_mm, expected_hc_mm)``,
        or ``None`` if the CSV is not found.
    """
    if not _HC18_CSV.exists():
        return None

    df = pd.read_csv(_HC18_CSV)
    return {
        str(row["filename"]): (
            float(row["pixel size(mm)"]),
            float(row["head circumference (mm)"]),
        )
        for _, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the post-processing and HC measurement validation."""
    model_name = input("Model (unet / yolov8): ").strip().lower()

    # Instantiate the appropriate segmenter and resolve its validation image path.
    if model_name == "unet":
        img_dir = here("data/preprocessed/fastai/images/val/")
        segmenter = UNetSegmenter(MODELS["unet"])
    elif model_name == "yolov8":
        img_dir = here("data/preprocessed/yolo/images/val/")
        segmenter = YOLOv8Segmenter(MODELS["yolov8"])
    else:
        raise ValueError(f"Unknown model '{model_name}'. Expected 'unet' or 'yolov8'.")

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Validation image directory not found at {img_dir}. "
            "Run preprocess.py first."
        )
    if not segmenter.available:
        raise FileNotFoundError(
            f"Model weights not found at {segmenter.spec.weights_path}."
        )

    # Load HC18 ground-truth data.
    hc18_data = load_hc18_data()
    if not hc18_data:
        print("HC18 ground-truth CSV not found. Cannot validate HC measurement.")
        return
    print("HC18 ground-truth data loaded.\n")

    # Initialise the ONNX session explicitly so session load time is excluded
    # from per-image timing measurements.
    print(f"Loading {segmenter.name} ONNX model from: {segmenter.spec.weights_path}…")
    segmenter._ensure_session()
    print("Model loaded.\n")

    # Select a random subset of validation images.
    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    num_test = min(10, len(all_images))
    test_images = random.sample(all_images, num_test)

    print("=" * 70)
    print(f"POST-PROCESSING & HC MEASUREMENT VALIDATION  (n={num_test})")
    print("=" * 70)

    results: list[dict] = []
    hc_errors: list[float] = []

    for img_name in test_images:
        if img_name not in hc18_data:
            print(f"  Skipping {img_name}: no HC18 ground-truth entry")
            continue

        pixel_size, expected_hc = hc18_data[img_name]
        img_rgb = load_image_rgb(img_dir / img_name)

        # Full inference pipeline: raw image → binary skull mask.
        seg_result = segmenter.predict(img_rgb)

        # Geometric pipeline: mask → largest contour → fitted ellipse → HC.
        hc_result = measure_hc(seg_result.mask, pixel_size)

        predicted_hc: float | None = None
        hc_error: float | None = None

        if hc_result is not None:
            predicted_hc = hc_result.hc_mm
            hc_error = abs(predicted_hc - expected_hc)
            hc_errors.append(hc_error)

        results.append(
            {
                "image": img_name,
                "predicted_hc": predicted_hc,
                "expected_hc": expected_hc,
                "hc_error": hc_error,
            }
        )

    # Per-image result table
    print("\nPer-Image Results:")
    print("-" * 70)
    print(f"{'Image':<20} {'Pred. HC':>10} {'Exp. HC':>10} {'Error':>10}")
    print("-" * 70)
    for r in results:
        pred_str = f"{r['predicted_hc']:.1f}" if r["predicted_hc"] is not None else "N/A"
        exp_str = f"{r['expected_hc']:.1f}"
        err_str = f"{r['hc_error']:.1f}" if r["hc_error"] is not None else "N/A"
        print(f"{r['image']:<20} {pred_str:>10} {exp_str:>10} {err_str:>10}")
    print("-" * 70)

    # Aggregate statistics
    if hc_errors:
        mean_err = np.mean(hc_errors)
        std_err = np.std(hc_errors)
        print("\nSummary Statistics:")
        print("-" * 70)
        print(f"{'Metric':<20} {'Mean':>10} {'Std Dev':>12}")
        print("-" * 70)
        print(f"{'HC Error (mm)':<20} {mean_err:>10.1f} {std_err:>12.1f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
