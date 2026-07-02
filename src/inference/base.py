"""
base.py
=======
Shared plumbing for ONNX-backed segmentation models.

* ``SegmentationResult`` — the common contract every model returns: a binary
  skull mask at the *original* image resolution, plus timing/confidence.
* ``Segmenter`` — an abstract base that owns the ONNX Runtime session and a few
  numpy helpers (letterbox, sigmoid). Concrete models implement ``predict``.

ONNX Runtime and OpenCV are imported lazily so this module stays importable in
environments where only the pure-Python maths is exercised (e.g. unit tests).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from config import ModelSpec


# --------------------------------------------------------------------------- #
# Result contract
# --------------------------------------------------------------------------- #
@dataclass
class SegmentationResult:
    """A model's prediction for one image."""

    mask: np.ndarray            # HxW uint8 in {0, 1}, original resolution
    inference_ms: float         # wall-clock time inside session.run
    model_name: str
    found: bool                 # True if a plausible skull was segmented
    confidence: float | None = None


# --------------------------------------------------------------------------- #
# numpy helpers (shared)
# --------------------------------------------------------------------------- #
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    color: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to ``new_shape`` (h, w).

    Returns the padded image, the scale ratio and the (left, top) padding so a
    predicted mask can be mapped back to the original frame.
    """
    import cv2

    h0, w0 = image.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / h0, new_w / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = new_w - nw, new_h - nh
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(color, color, color),
    )
    return padded, r, (left, top)


# --------------------------------------------------------------------------- #
# Abstract segmenter
# --------------------------------------------------------------------------- #
class Segmenter(ABC):
    """Base class owning an ONNX Runtime session for one exported model."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._session = None
        self._input_name: str | None = None
        self._output_names: list[str] | None = None

    # -- session lifecycle ------------------------------------------------- #
    @property
    def available(self) -> bool:
        """True when the weights file exists on disk."""
        return Path(self.spec.weights_path).exists()

    @property
    def name(self) -> str:
        return self.spec.display_name

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        if not self.available:
            raise FileNotFoundError(
                f"Weights not found for {self.spec.display_name}: "
                f"{self.spec.weights_path}"
            )
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.spec.weights_path), sess_options=so, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def _run(self, tensor: np.ndarray) -> tuple[list[np.ndarray], float]:
        """Execute the session and return (outputs, elapsed_ms)."""
        self._ensure_session()
        t0 = time.perf_counter()
        outputs = self._session.run(self._output_names, {self._input_name: tensor})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return outputs, elapsed_ms

    # -- preprocessing helper --------------------------------------------- #
    def _preprocess_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Resize + normalize to a plain NCHW tensor (used by U-Net).

        YOLOv8 overrides this with letterboxing.
        """
        import cv2

        h, w = self.spec.input_size
        if self.spec.channels == 1:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
            arr = resized.astype(np.float32)[None, ...]          # (1, H, W)
        else:
            resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            arr = resized.astype(np.float32).transpose(2, 0, 1)   # (3, H, W)

        if self.spec.normalize == "standard":
            mean = np.array(self.spec.mean, dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(self.spec.std, dtype=np.float32).reshape(-1, 1, 1)
            arr = (arr / 255.0 - mean) / std
        else:  # "scale"
            arr = arr / 255.0

        return arr[None, ...].astype(np.float32)                  # (1, C, H, W)

    # -- contract ---------------------------------------------------------- #
    @abstractmethod
    def predict(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Segment the fetal skull in an RGB uint8 image."""
        raise NotImplementedError
