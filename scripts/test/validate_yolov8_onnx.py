"""
validate_yolov8_onnx.py
=======================
Benchmark and accuracy evaluation for the exported YOLOv8-seg ONNX model.

Evaluation protocol
--------------------
1. Instantiate the ``YOLOv8Segmenter`` wrapper and report the ONNX input
   tensor specification.
2. Measure CPU inference latency over 50 benchmark runs (10 warm-up runs
   excluded) and flag whether the ≤200 ms budget is met.
3. Sample up to 10 validation images, run end-to-end inference (including
   letterboxing, NMS-free head decoding, and mask assembly), compare the
   predicted mask against the preprocessed ground-truth mask, and report
   per-image and aggregate metrics:

   * IoU (Jaccard index)
   * Dice coefficient (F₁ score for segmentation)
   * Pixel accuracy

The ground-truth masks are loaded from the FastAI split directory, which is
shared between both model families and avoids duplication.

Usage
-----
    python scripts/test/validate_yolov8_onnx.py
"""

from __future__ import annotations

import os
import random
import sys
import time

import cv2
import numpy as np
from pyprojroot import here

sys.path.insert(0, str(here("src")))
from config import MODELS
from inference.yolov8_onnx import YOLOv8Segmenter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMG_PATH = here("data/preprocessed/yolo/images/val/")
# Ground-truth masks are produced by preprocess.py under the FastAI tree;
# they are identical for both model formats.
MASK_PATH = here("data/preprocessed/fastai/masks/val/")
MODEL_PATH = here("models/yolov8_hc.onnx")

# Benchmark parameters
_WARMUP_RUNS = 10
_BENCHMARK_RUNS = 50
_LATENCY_BUDGET_MS = 200.0


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Intersection over Union (Jaccard index) for two binary masks.

    Parameters
    ----------
    pred_mask:
        Binary prediction mask with values 0 or 1, shape (H, W).
    gt_mask:
        Binary ground-truth mask with values 0 or 1, shape (H, W).

    Returns
    -------
    float
        IoU score in [0, 1].  Returns 1.0 when both masks are empty.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def calculate_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute the Dice coefficient (F₁ score) for two binary masks.

    Parameters
    ----------
    pred_mask:
        Binary prediction mask with values 0 or 1, shape (H, W).
    gt_mask:
        Binary ground-truth mask with values 0 or 1, shape (H, W).

    Returns
    -------
    float
        Dice score in [0, 1].  Returns 1.0 when both masks are empty.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_pixels = pred_mask.sum() + gt_mask.sum()
    if total_pixels == 0:
        return 1.0 if intersection == 0 else 0.0
    return float((2.0 * intersection) / total_pixels)


def calculate_pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute the fraction of correctly classified pixels.

    Parameters
    ----------
    pred_mask:
        Binary prediction mask with values 0 or 1, shape (H, W).
    gt_mask:
        Binary ground-truth mask with values 0 or 1, shape (H, W).

    Returns
    -------
    float
        Pixel accuracy in [0, 1].
    """
    return float((pred_mask == gt_mask).sum() / pred_mask.size)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def load_image_rgb(img_path: os.PathLike) -> np.ndarray:
    """Load an image from disk as an RGB uint8 array.

    The YOLOv8Segmenter wrapper accepts raw RGB images and handles all
    model-specific preprocessing (letterboxing, normalization, tensor packing)
    internally inside ``predict``.

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


def load_ground_truth_mask(
    mask_path: os.PathLike,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Load and binarize a ground-truth segmentation mask.

    Resizes the mask to ``target_shape`` using nearest-neighbour interpolation
    to preserve binary values without introducing interpolation artefacts.

    Parameters
    ----------
    mask_path:
        Path to the grayscale PNG mask file (values 0 or 255).
    target_shape:
        Target spatial dimensions as (height, width).  Note: OpenCV uses
        (width, height) ordering for ``resize``; this function handles the
        conversion internally.

    Returns
    -------
    numpy.ndarray
        Binary uint8 mask of shape ``target_shape`` with values 0 or 1.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(
        mask,
        (target_shape[1], target_shape[0]),  # cv2.resize takes (w, h)
        interpolation=cv2.INTER_NEAREST,
    )
    return (mask > 128).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

# Guard: verify all required artefacts are present before loading the model.
for _path, _desc in [
    (IMG_PATH, "validation image directory"),
    (MASK_PATH, "ground-truth mask directory"),
    (MODEL_PATH, "YOLOv8 ONNX model file"),
]:
    if not _path.exists():
        raise FileNotFoundError(
            f"{_desc.capitalize()} not found at {_path}. "
            "Ensure preprocessing has completed and model weights are in place."
        )

# Load the model through the shared segmenter wrapper so that the same
# preprocessing and decoding path used in production is exercised here.
print(f"Loading YOLOv8 ONNX model from: {MODEL_PATH}…")
_t_load = time.time()
segmenter = YOLOv8Segmenter(MODELS["yolov8"])
segmenter._ensure_session()
session = segmenter._session
print(f"Model loaded in {time.time() - _t_load:.3f} s\n")

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input name  : {input_name}")
print(f"Input shape : {input_shape}\n")

# Select a random subset of validation images.
all_images = sorted([f for f in os.listdir(IMG_PATH) if f.endswith(".png")])
num_test_images = min(10, len(all_images))
if num_test_images < 10:
    print(f"Warning: only {num_test_images} images available (requested 10)\n")
test_images = random.sample(all_images, num_test_images)

# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------
print("=" * 70)
print("LATENCY BENCHMARK")
print("=" * 70)

first_img = load_image_rgb(IMG_PATH / test_images[0])

# Warm-up pass to prime ONNX Runtime graph optimizations and OS I/O caches.
result = segmenter.predict(first_img)
print(f"Prediction mask shape: {result.mask.shape}\n")

for _ in range(_WARMUP_RUNS):
    segmenter.predict(first_img)

latencies: list[float] = []
for _ in range(_BENCHMARK_RUNS):
    t0 = time.perf_counter()
    segmenter.predict(first_img)
    latencies.append(time.perf_counter() - t0)

mean_ms = np.mean(latencies) * 1000
p95_ms = np.percentile(latencies, 95) * 1000
print(f"Mean CPU latency : {mean_ms:.1f} ms  |  P95 : {p95_ms:.1f} ms")

if mean_ms < _LATENCY_BUDGET_MS:
    print(f"Latency budget met: mean < {_LATENCY_BUDGET_MS:.0f} ms\n")
else:
    print(f"Latency budget missed: mean exceeds {_LATENCY_BUDGET_MS:.0f} ms\n")

# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"ACCURACY METRICS  (n={num_test_images} images)")
print("=" * 70)

iou_scores: list[float] = []
dice_scores: list[float] = []
pixel_acc_scores: list[float] = []
results: list[dict] = []

for img_name in test_images:
    img_path = IMG_PATH / img_name
    mask_path = MASK_PATH / img_name

    if not mask_path.exists():
        print(f"  Skipping {img_name}: ground-truth mask not found")
        continue

    img = load_image_rgb(img_path)
    gt_mask = load_ground_truth_mask(mask_path, img.shape[:2])

    # ``predict`` runs the full inference-to-mask pipeline, returning a
    # binary mask at the original image resolution.
    result = segmenter.predict(img)
    pred_mask = result.mask

    iou = calculate_iou(pred_mask, gt_mask)
    dice = calculate_dice(pred_mask, gt_mask)
    pixel_acc = calculate_pixel_accuracy(pred_mask, gt_mask)

    iou_scores.append(iou)
    dice_scores.append(dice)
    pixel_acc_scores.append(pixel_acc)
    results.append({"image": img_name, "iou": iou, "dice": dice, "pixel_acc": pixel_acc})

# Per-image table
if not results:
    print("No valid image-mask pairs found for evaluation.")
else:
    print("\nPer-Image Results:")
    print("-" * 70)
    print(f"{'Image':<20} {'IoU':>10} {'Dice':>10} {'Pixel Acc':>12}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['image']:<20} {r['iou']:>10.3f} {r['dice']:>10.3f}"
            f" {r['pixel_acc']:>12.3f}"
        )
    print("-" * 70)

    # Aggregate statistics
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    mean_pixel_acc = np.mean(pixel_acc_scores)

    print("\nSummary Statistics:")
    print("-" * 70)
    print(f"{'Metric':<20} {'Mean':>10} {'Std Dev':>12}")
    print("-" * 70)
    print(f"{'IoU (Jaccard)':<20} {mean_iou:>10.3f} {np.std(iou_scores):>12.3f}")
    print(f"{'Dice Coefficient':<20} {mean_dice:>10.3f} {np.std(dice_scores):>12.3f}")
    print(f"{'Pixel Accuracy':<20} {mean_pixel_acc:>10.3f} {np.std(pixel_acc_scores):>12.3f}")
    print("=" * 70)

    if mean_iou > 0.7 and mean_dice > 0.8:
        print("\nModel accuracy: GOOD  (IoU > 0.7, Dice > 0.8)")
    elif mean_iou > 0.5 and mean_dice > 0.6:
        print("\nModel accuracy: ACCEPTABLE  (IoU > 0.5, Dice > 0.6)")
    else:
        print("\nModel accuracy: LOW  (IoU < 0.5 or Dice < 0.6)")