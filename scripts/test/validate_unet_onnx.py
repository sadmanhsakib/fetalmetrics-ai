"""
validate_unet_onnx.py
=====================
Benchmark and accuracy evaluation for the exported U-Net ResNet34 ONNX model.

Evaluation protocol
--------------------
1. Load the model and report the ONNX input tensor specification.
2. Measure CPU inference latency over 50 benchmark runs (10 warm-up runs
   excluded) and flag whether the ≤200 ms budget is met.
3. Sample up to 10 validation images, run inference, compare the predicted
   mask against the preprocessed ground-truth mask, and report per-image and
   aggregate metrics:

   * IoU (Jaccard index)
   * Dice coefficient (F₁ score for segmentation)
   * Pixel accuracy

Usage
-----
    python scripts/test/validate_unet_onnx.py
"""

from __future__ import annotations

import os
import random
import time

import cv2
import numpy as np
import onnxruntime as ort
from pyprojroot import here

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMG_PATH = here("data/preprocessed/fastai/images/val/")
MASK_PATH = here("data/preprocessed/fastai/masks/val/")
MODEL_PATH = here("models/unet_hc.onnx")

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
        IoU score in [0, 1].  Returns 1.0 when both masks are empty
        (perfect agreement on a trivially empty image).
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
    correct = (pred_mask == gt_mask).sum()
    return float(correct / pred_mask.size)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def preprocess_image(img_path: os.PathLike, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load and preprocess an image to the U-Net ONNX input tensor format.

    The pipeline mirrors the training-time normalization:
    * Resize to ``target_size`` (H × W).
    * Scale to [0, 1].
    * Apply ImageNet channel-wise mean and standard deviation.
    * Transpose to NCHW layout.

    Parameters
    ----------
    img_path:
        Path to the input image file.
    target_size:
        Target spatial dimensions as (width, height) for ``cv2.resize``.

    Returns
    -------
    numpy.ndarray
        Float32 tensor of shape (1, 3, H, W), ready for ONNX Runtime.
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)

    img = img.astype(np.float32) / 255.0

    # ImageNet channel statistics — must match the training-time normalization.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))   # (H, W, C) → (C, H, W)
    img = np.expand_dims(img, axis=0)     # (C, H, W) → (1, C, H, W)

    return img.astype(np.float32)


def load_ground_truth_mask(
    mask_path: os.PathLike,
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Load and binarize a ground-truth segmentation mask.

    Parameters
    ----------
    mask_path:
        Path to the mask image file (grayscale PNG, values 0 or 255).
    target_size:
        Target spatial dimensions as (width, height) for nearest-neighbour
        resize.

    Returns
    -------
    numpy.ndarray
        Binary uint8 mask of shape (H, W) with values 0 or 1.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return (mask > 128).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

# Guard: verify all required artefacts are present before loading the model.
for _path, _desc in [
    (IMG_PATH, "validation image directory"),
    (MASK_PATH, "ground-truth mask directory"),
    (MODEL_PATH, "U-Net ONNX model file"),
]:
    if not _path.exists():
        raise FileNotFoundError(
            f"{_desc.capitalize()} not found at {_path}. "
            "Ensure preprocessing has completed and model weights are in place."
        )

# Load the ONNX model.
print(f"Loading U-Net ONNX model from: {MODEL_PATH}…")
_t_load = time.time()
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
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

first_tensor = preprocess_image(IMG_PATH / test_images[0])

# Warm-up pass to prime ONNX Runtime graph optimizations.
outputs = session.run(None, {input_name: first_tensor})
print(f"Output shape: {outputs[0].shape}\n")

for _ in range(_WARMUP_RUNS):
    session.run(None, {input_name: first_tensor})

latencies: list[float] = []
for _ in range(_BENCHMARK_RUNS):
    t0 = time.perf_counter()
    session.run(None, {input_name: first_tensor})
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

    img_tensor = preprocess_image(img_path)
    gt_mask = load_ground_truth_mask(mask_path)

    # U-Net ONNX output: (1, 2, H, W) logits — argmax over the class axis.
    outputs = session.run(None, {input_name: img_tensor})
    pred_mask = np.argmax(outputs[0], axis=1)[0]

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

    # Qualitative assessment against typical HC18 benchmarks.
    if mean_iou > 0.7 and mean_dice > 0.8:
        print("\nModel accuracy: GOOD  (IoU > 0.7, Dice > 0.8)")
    elif mean_iou > 0.5 and mean_dice > 0.6:
        print("\nModel accuracy: ACCEPTABLE  (IoU > 0.5, Dice > 0.6)")
    else:
        print("\nModel accuracy: LOW  (IoU < 0.5 or Dice < 0.6)")
