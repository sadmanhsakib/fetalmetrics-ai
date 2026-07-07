"""Verification script for U-Net ResNet34 ONNX Model with Accuracy Metrics."""
import os
import random
import time

import cv2
import numpy as np
import onnxruntime as ort

from pyprojroot import here

# ============================================================================
# Metric Calculation Functions
# ============================================================================

def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union (IoU) / Jaccard Index.

    Args:
        pred_mask: Binary prediction mask (0 or 1)
        gt_mask: Binary ground truth mask (0 or 1)

    Returns:
        IoU score (float between 0 and 1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union

def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice Coefficient (F1-score for segmentation).

    Args:
        pred_mask: Binary prediction mask (0 or 1)
        gt_mask: Binary ground truth mask (0 or 1)

    Returns:
        Dice score (float between 0 and 1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_pixels = pred_mask.sum() + gt_mask.sum()

    if total_pixels == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2.0 * intersection) / total_pixels

def calculate_pixel_accuracy(pred_mask, gt_mask):
    """Calculate pixel-level accuracy.

    Args:
        pred_mask: Binary prediction mask (0 or 1)
        gt_mask: Binary ground truth mask (0 or 1)

    Returns:
        Pixel accuracy (float between 0 and 1)
    """
    correct = (pred_mask == gt_mask).sum()
    total = pred_mask.size
    return correct / total

def preprocess_image(img_path, target_size=(256, 256)):
    """Load and preprocess an image for U-Net inference.

    Args:
        img_path: Path to image file
        target_size: Target (height, width) for resizing

    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # Transpose to (C, H, W) and add batch dimension
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)

def load_ground_truth_mask(mask_path, target_size=(256, 256)):
    """Load and binarize a ground truth mask.

    Args:
        mask_path: Path to mask file
        target_size: Target (height, width) for resizing

    Returns:
        Binary mask (H, W) with values 0 or 1
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Binarize: threshold at mid-value (128)
    mask = (mask > 128).astype(np.uint8)

    return mask

# ============================================================================
# Main Validation Script
# ============================================================================

# 1. Setup paths
IMG_PATH = here("data/preprocessed/fastai/images/val/")
MASK_PATH = here("data/preprocessed/fastai/masks/val/")
MODEL_PATH = here("models/unet_hc.onnx")

if not IMG_PATH.exists():
    raise FileNotFoundError(
        f"Image directory not found at {IMG_PATH}. "
        "Make sure it is downloaded and placed in the right directory."
    )
if not MASK_PATH.exists():
    raise FileNotFoundError(
        f"Mask directory not found at {MASK_PATH}. "
        "Make sure ground truth masks are available."
    )
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Make sure it is downloaded and placed in the right directory."
    )

# 2. Load model
print(f"Loading U-Net ONNX model from: {MODEL_PATH}...")
start_time = time.time()
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
print(f"Model loaded in {time.time() - start_time:.3f}s\n")

# 3. Retrieve input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input Name:  {input_name}")
print(f"Input Shape: {input_shape}\n")

# 4. Select random images for accuracy evaluation
all_images = sorted([f for f in os.listdir(IMG_PATH) if f.endswith('.png')])
num_test_images = min(10, len(all_images))

if num_test_images < 10:
    print(f"⚠️  Warning: Only {num_test_images} images available (requested 10)\n")

test_images = random.sample(all_images, num_test_images)

# 5. Benchmark speed on first image
print("=" * 70)
print("LATENCY BENCHMARK")
print("=" * 70)

first_img_path = IMG_PATH / test_images[0]
first_img_tensor = preprocess_image(first_img_path)

# Warm-up pass
outputs = session.run(None, {input_name: first_img_tensor})
print(f"Output shape: {outputs[0].shape}\n")

# Benchmark
WARMUP_RUNS = 10
BENCHMARK_RUNS = 50

for _ in range(WARMUP_RUNS):
    session.run(None, {input_name: first_img_tensor})

latencies = []
for _ in range(BENCHMARK_RUNS):
    t0 = time.perf_counter()
    session.run(None, {input_name: first_img_tensor})
    latencies.append(time.perf_counter() - t0)

mean_latency_ms = np.mean(latencies) * 1000
p95_latency_ms = np.percentile(latencies, 95) * 1000
print(f"Mean CPU Latency: {mean_latency_ms:.1f}ms  |  P95: {p95_latency_ms:.1f}ms")

if mean_latency_ms < 200:
    print("✅ Target constraint met: CPU latency is under 200ms.\n")
else:
    print("⚠️  Target constraint missed: CPU latency is above 200ms.\n")

# 6. Calculate accuracy metrics
print("=" * 70)
print(f"ACCURACY METRICS (n={num_test_images} images)")
print("=" * 70)

iou_scores = []
dice_scores = []
pixel_acc_scores = []
results = []

for img_name in test_images:
    img_path = IMG_PATH / img_name
    mask_path = MASK_PATH / img_name

    # Check if ground truth exists
    if not mask_path.exists():
        print(f"⚠️  Skipping {img_name}: ground truth mask not found")
        continue

    # Load and preprocess
    img_tensor = preprocess_image(img_path)
    gt_mask = load_ground_truth_mask(mask_path)

    # Run inference
    outputs = session.run(None, {input_name: img_tensor})
    logits = outputs[0]  # shape: (1, 2, 256, 256)
    pred_mask = np.argmax(logits, axis=1)[0]  # shape: (256, 256)

    # Calculate metrics
    iou = calculate_iou(pred_mask, gt_mask)
    dice = calculate_dice(pred_mask, gt_mask)
    pixel_acc = calculate_pixel_accuracy(pred_mask, gt_mask)

    iou_scores.append(iou)
    dice_scores.append(dice)
    pixel_acc_scores.append(pixel_acc)

    results.append({
        'image': img_name,
        'iou': iou,
        'dice': dice,
        'pixel_acc': pixel_acc
    })

# 7. Display results
if not results:
    print("❌ No valid image-mask pairs found for evaluation")
else:
    print("\nPer-Image Results:")
    print("-" * 70)
    print(f"{'Image':<20} {'IoU':>10} {'Dice':>10} {'Pixel Acc':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['image']:<20} {r['iou']:>10.3f} {r['dice']:>10.3f} {r['pixel_acc']:>12.3f}")

    print("-" * 70)

    # Summary statistics
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    mean_pixel_acc = np.mean(pixel_acc_scores)
    std_pixel_acc = np.std(pixel_acc_scores)

    print("\nSummary Statistics:")
    print("-" * 70)
    print(f"{'Metric':<20} {'Mean':>10} {'Std Dev':>12}")
    print("-" * 70)
    print(f"{'IoU (Jaccard)':<20} {mean_iou:>10.3f} {std_iou:>12.3f}")
    print(f"{'Dice Coefficient':<20} {mean_dice:>10.3f} {std_dice:>12.3f}")
    print(f"{'Pixel Accuracy':<20} {mean_pixel_acc:>10.3f} {std_pixel_acc:>12.3f}")
    print("=" * 70)

    # Final assessment
    if mean_iou > 0.7 and mean_dice > 0.8:
        print("\n✅ Model accuracy is GOOD (IoU > 0.7, Dice > 0.8)")
    elif mean_iou > 0.5 and mean_dice > 0.6:
        print("\n⚠️  Model accuracy is ACCEPTABLE (IoU > 0.5, Dice > 0.6)")
    else:
        print("\n❌ Model accuracy is LOW (IoU < 0.5 or Dice < 0.6)")
