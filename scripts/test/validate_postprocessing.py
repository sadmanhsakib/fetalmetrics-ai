"""Validation script for post-processing pipeline and fetal circumference measurement."""
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd

from pyprojroot import here

# Add src to path
sys.path.insert(0, str(here("src")))

from config import MODELS
from inference.unet_onnx import UNetSegmenter
from inference.yolov8_onnx import YOLOv8Segmenter
from postprocess.ellipse import measure_hc


def load_image_rgb(img_path) -> np.ndarray:
    """Load an image as an RGB uint8 array (H, W, 3).

    Both UNetSegmenter and YOLOv8Segmenter accept raw RGB images and handle
    all model-specific preprocessing (resize, normalization, letterboxing)
    internally via their ``predict`` method.
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_hc18_data():
    """Load HC18 data for expected circumferences and pixel sizes."""
    hc18_csv_path = here("data/raw/training_set_pixel_size_and_HC.csv")
    if not hc18_csv_path.exists():
        return None
    df = pd.read_csv(hc18_csv_path)
    hc18_data = {}
    for _, row in df.iterrows():
        filename = row['filename']
        pixel_size = row['pixel size(mm)']
        expected_hc = row['head circumference (mm)']
        hc18_data[filename] = (pixel_size, expected_hc)
    return hc18_data


def main():
    model_name = input("Enter model name (unet/yolov8): ").strip().lower()

    # Setup paths and instantiate the appropriate segmenter
    if model_name == "unet":
        IMG_PATH = here("data/preprocessed/fastai/images/val/")
        segmenter = UNetSegmenter(MODELS["unet"])
    elif model_name == "yolov8":
        IMG_PATH = here("data/preprocessed/yolo/images/val/")
        segmenter = YOLOv8Segmenter(MODELS["yolov8"])
    else:
        raise ValueError("Invalid model name. Please enter 'unet' or 'yolov8'.")

    if not IMG_PATH.exists():
        raise FileNotFoundError(
            f"Image directory not found at {IMG_PATH}"
        )
    if not segmenter.available:
        raise FileNotFoundError(
            f"Model file not found at {segmenter.spec.weights_path}"
        )

    # Load HC18 data
    hc18_data = load_hc18_data()
    if not hc18_data:
        print("❌ HC18 ground truth data not found. Cannot validate circumference.")
        return
    print("✅ Loaded HC18 ground truth data\n")

    # Load model
    print(f"Loading {segmenter.name} ONNX model from: {segmenter.spec.weights_path}...")
    segmenter._ensure_session()
    print("Model loaded\n")

    # Select test images
    all_images = sorted([f for f in os.listdir(IMG_PATH) if f.endswith('.png')])
    num_test_images = min(10, len(all_images))
    test_images = random.sample(all_images, num_test_images)

    # Calculate circumference metrics
    print("=" * 70)
    print(f"POST-PROCESSING & CIRCUMFERENCE VALIDATION (n={num_test_images})")
    print("=" * 70)

    results = []
    hc_errors = []

    for img_name in test_images:
        if img_name not in hc18_data:
            print(f"⚠️  Skipping {img_name}: No HC18 data")
            continue

        img_path = IMG_PATH / img_name
        pixel_size, expected_hc = hc18_data[img_name]

        # Run inference — each segmenter handles its own preprocessing and
        # output decoding, returning a binary mask at the original resolution.
        img_rgb = load_image_rgb(img_path)
        seg_result = segmenter.predict(img_rgb)
        pred_mask = seg_result.mask

        # Use post-processing to measure HC
        hc_result = measure_hc(pred_mask, pixel_size)

        predicted_hc = None
        hc_error = None
        if hc_result:
            predicted_hc = hc_result.hc_mm
            hc_error = abs(predicted_hc - expected_hc)
            hc_errors.append(hc_error)

        results.append({
            'image': img_name,
            'predicted_hc': predicted_hc,
            'expected_hc': expected_hc,
            'hc_error': hc_error
        })

    # Display results
    print("\nPer-Image Results:")
    print("-" * 70)
    print(f"{'Image':<20} {'Pred. HC':>10} {'Exp. HC':>10} {'Error':>10}")
    print("-" * 70)

    for r in results:
        pred_hc = f"{r['predicted_hc']:.1f}" if r['predicted_hc'] else "N/A"
        exp_hc = f"{r['expected_hc']:.1f}"
        hc_err = f"{r['hc_error']:.1f}" if r['hc_error'] else "N/A"
        print(f"{r['image']:<20} {pred_hc:>10} {exp_hc:>10} {hc_err:>10}")

    print("-" * 70)

    # Summary statistics
    if hc_errors:
        mean_hc_error = np.mean(hc_errors)
        std_hc_error = np.std(hc_errors)
        print("\nSummary Statistics:")
        print("-" * 70)
        print(f"{'Metric':<20} {'Mean':>10} {'Std Dev':>12}")
        print("-" * 70)
        print(f"{'HC Error (mm)':<20} {mean_hc_error:>10.1f} {std_hc_error:>12.1f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
