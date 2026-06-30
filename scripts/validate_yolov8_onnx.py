"""ONNX Model Inference Verification Script."""
import os
import random
import time

import cv2
import numpy as np
import onnxruntime as ort

from pyprojroot import here

# 1. Setup paths
IMG_PATH = here("data/preprocessed/yolo/images/train/")
MODEL_PATH = here("models/yolov8s-seg.onnx")

if not IMG_PATH.exists():
    raise FileNotFoundError(f"Image directory not found at {IMG_PATH}. Make sure it is downloaded and placed in the right directory. ")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure it is downloaded and placed in the right directory. ")

# 2. Initialize ONNX runtime session (using CPU provider for compatibility)
print(f"Loading ONNX model from: {MODEL_PATH}...")
start_time = time.time()
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
load_time = time.time() - start_time
print(f"Model loaded successfully in {load_time:.3f}s")

# 3. Retrieve input details
input_details = session.get_inputs()
input_name = input_details[0].name
input_shape = input_details[0].shape
print(f"Input Name:  {input_name}")
print(f"Input Shape: {input_shape}") # Expected: [1, 3, 640, 640]

# 4.loading one random image from the training set for testing
img_name = random.choice(os.listdir(IMG_PATH))
random_img = cv2.imread(os.path.join(IMG_PATH, img_name))
random_img = cv2.cvtColor(random_img, cv2.COLOR_BGR2RGB)
random_img = cv2.resize(random_img, (640, 640))

# normalizing to [0, 1]
random_img = random_img.astype(np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
random_img = (random_img - mean) / std

# transposing to (1, 3, 640, 640)
random_img = np.transpose(random_img, (2, 0, 1))
random_img = np.expand_dims(random_img, axis=0)
random_img = random_img.astype(np.float32)

# 5. Warm-up pass
_ = session.run(None, {input_name: random_img})

# 6. Benchmark speed
WARMUP_RUNS = 10
BENCHMARK_RUNS = 50  # More runs → lower variance in the reported mean

for _ in range(WARMUP_RUNS):
    session.run(None, {input_name: random_img})

latencies = []
for _ in range(BENCHMARK_RUNS):
    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: random_img})
    latencies.append(time.perf_counter() - t0)

mean_latency_ms = np.mean(latencies) * 1000
print(f"Mean CPU Inference Latency: {mean_latency_ms:.1f}ms")
print(f"Number of Output Tensors: {len(outputs)}")

# Verify target constraint
if mean_latency_ms < 200:
    print("✅ Target constraint met: CPU latency is under 200ms.")
else:
    print("⚠️ Target constraint missed: CPU latency is above 200ms.")