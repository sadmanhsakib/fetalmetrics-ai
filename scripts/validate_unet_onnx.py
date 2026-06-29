"""Verification script for U-Net ResNet34 ONNX Model."""
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from pyprojroot import here

# 1. Setup paths
MODEL_PATH = here("models/unet-resnet34.onnx")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure it is downloaded.")

# 2. Load model
print(f"Loading U-Net ONNX model from: {MODEL_PATH}...")
start_time = time.time()
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
print(f"Model loaded in {time.time() - start_time:.3f}s")

# 3. Retrieve input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input Name:  {input_name}")
print(f"Input Shape: {input_shape}") # Expected: ['batch_size', 3, 256, 256]

# 4. Create dummy input data matching shape [1, 3, 256, 256]
dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)

# 5. Warm-up pass
outputs = session.run(None, {input_name: dummy_input})
print(f"Output shape: {outputs[0].shape}") # Expected: (1, 2, 256, 256)

# 6. Benchmark speed
WARMUP_RUNS = 10
BENCHMARK_RUNS = 50  # More runs → lower variance in the reported mean

for _ in range(WARMUP_RUNS):
    session.run(None, {input_name: dummy_input})

latencies = []
for _ in range(BENCHMARK_RUNS):
    t0 = time.perf_counter()
    session.run(None, {input_name: dummy_input})
    latencies.append(time.perf_counter() - t0)

mean_latency_ms = np.mean(latencies) * 1000
p95_latency_ms  = np.percentile(latencies, 95) * 1000
print(f"Mean CPU Latency: {mean_latency_ms:.1f}ms  |  P95: {p95_latency_ms:.1f}ms")

# Parse output (Complication #4: Argmax over channels)
logits = outputs[0]  # shape: (1, 2, 256, 256)
pred_mask = np.argmax(logits, axis=1)[0]  # shape: (256, 256)
print(f"Parsed predicted mask values: {np.unique(pred_mask)} (Expected: [0 1] or subset)")

# Verify target constraint
if mean_latency_ms < 200:
    print("✅ Target constraint met: CPU latency is under 200ms.")
else:
    print("⚠️ Target constraint missed: CPU latency is above 200ms.")