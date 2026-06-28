"""ONNX Model Inference Verification Script."""
import time
import numpy as np
import onnxruntime as ort
from pyprojroot import here

# 1. Setup paths
MODEL_PATH = here("models/yolov8s-seg.onnx")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Did you download it from Kaggle?")

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

# 4. Create dummy input image (matching shape 1x3x640x640)
dummy_image = np.random.randn(1, 3, 640, 640).astype(np.float32)

# 5. Warm-up pass
_ = session.run(None, {input_name: dummy_image})

# 6. Benchmark speed
runs = 20
latencies = []
for _ in range(runs):
    t0 = time.time()
    outputs = session.run(None, {input_name: dummy_image})
    latencies.append(time.time() - t0)

mean_latency_ms = np.mean(latencies) * 1000
print(f"Average CPU Inference Latency: {mean_latency_ms:.1f}ms")
print(f"Number of Output Tensors: {len(outputs)}")

# Verify target constraint
if mean_latency_ms < 200:
    print("✅ Target constraint met: CPU latency is under 200ms.")
else:
    print("⚠️ Target constraint missed: CPU latency is above 200ms.")