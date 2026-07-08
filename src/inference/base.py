"""
base.py
=======
Shared plumbing for ONNX-backed segmentation models.

Provides two public contracts:

* ``SegmentationResult`` — the common return type for every model's
  ``predict`` call: a binary skull mask at the **original** image resolution,
  plus inference timing and an optional confidence score.

* ``Segmenter`` — abstract base class that owns the ONNX Runtime session
  lifecycle and provides reusable NumPy helpers (letterbox, sigmoid,
  image preprocessing).  Concrete model wrappers (``YOLOv8Segmenter``,
  ``UNetSegmenter``) subclass ``Segmenter`` and implement ``predict``.

ONNX Runtime and OpenCV are imported **lazily** inside the methods that
require them so that the module remains importable in pure-Python environments
(e.g. unit tests) where neither dependency is installed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import ModelSpec


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    """Output of a single model inference pass.

    Attributes
    ----------
    mask:
        Binary segmentation mask of shape (H, W) in uint8 ``{0, 1}``,
        at the **original** input image resolution.
    inference_ms:
        Wall-clock time elapsed during ``session.run``, in milliseconds.
    model_name:
        Human-readable model name (from ``ModelSpec.display_name``).
    found:
        ``True`` when the model returned a non-empty, plausible skull mask.
    confidence:
        Model-specific confidence proxy, or ``None`` when not available.
        For YOLOv8, this is the detection confidence of the highest-scoring
        anchor.  For U-Net, it is the mean predicted probability over the
        thresholded foreground region.
    """

    mask: np.ndarray            # (H, W) uint8, values in {0, 1}
    inference_ms: float
    model_name: str
    found: bool
    confidence: float | None = None


# ---------------------------------------------------------------------------
# NumPy helpers (shared across model families)
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the logistic sigmoid element-wise to ``x``.

    Uses a numerically stable formulation that avoids overflow in the positive
    half and underflow in the negative half of the input domain.

    Parameters
    ----------
    x:
        Input array of arbitrary shape (typically model logits).

    Returns
    -------
    numpy.ndarray
        Array of the same shape with values in (0, 1).
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    color: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize an image to ``new_shape`` while preserving its aspect ratio.

    The image is scaled so that the larger dimension matches the corresponding
    target dimension exactly, then the shorter dimension is symmetrically
    padded with a constant fill value.  This is the standard YOLOv8
    pre-processing transform.

    Parameters
    ----------
    image:
        Input image of shape (H, W, C) or (H, W).
    new_shape:
        Target spatial dimensions as ``(height, width)``.
    color:
        Constant fill value for padding pixels (default 114, the YOLOv8
        convention for neutral gray).

    Returns
    -------
    padded:
        Padded image of shape ``(new_height, new_width, C)`` in the same
        dtype as ``image``.
    ratio:
        Uniform scale factor applied to both dimensions before padding.
        Multiply a coordinate in the padded frame by ``1 / ratio`` to recover
        the original-frame coordinate (before also subtracting the pad offset).
    (left, top):
        Number of padding columns and rows added to the left and top edges
        respectively.  Required to map predicted coordinates back to the
        original image frame.
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


# ---------------------------------------------------------------------------
# Abstract segmenter base class
# ---------------------------------------------------------------------------

class Segmenter(ABC):
    """Abstract base class for ONNX-backed skull segmentation models.

    Manages the ONNX Runtime session lifecycle (lazy initialisation, CUDA
    provider detection) and provides shared preprocessing utilities.
    Concrete subclasses only need to implement the ``predict`` method.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._session = None
        self._input_name: str | None = None
        self._output_names: list[str] | None = None

    # -- Session lifecycle --------------------------------------------------

    @property
    def available(self) -> bool:
        """``True`` when the ONNX weights file exists on disk."""
        return Path(self.spec.weights_path).exists()

    @property
    def name(self) -> str:
        """Human-readable model name from the ``ModelSpec``."""
        return self.spec.display_name

    def _ensure_session(self) -> None:
        """Lazily initialise the ONNX Runtime inference session.

        Called automatically by ``_run`` on the first inference request.
        Prefers the CUDA execution provider when available, falling back to
        CPU.  Raises ``FileNotFoundError`` when the weights file is absent to
        produce a clear error message rather than an opaque ONNX exception.
        """
        if self._session is not None:
            return

        if not self.available:
            raise FileNotFoundError(
                f"Weights not found for {self.spec.display_name}: "
                f"{self.spec.weights_path}"
            )

        import onnxruntime as ort

        # Prefer GPU acceleration when available; CPU is always the fallback.
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

    # -- Inference execution ------------------------------------------------

    def _run(self, tensor: np.ndarray) -> tuple[list[np.ndarray], float]:
        """Execute one forward pass through the ONNX session.

        Parameters
        ----------
        tensor:
            Pre-processed input tensor of shape (1, C, H, W) in float32.

        Returns
        -------
        outputs:
            List of raw output arrays from the model, in the order declared
            by the ONNX graph.
        elapsed_ms:
            Wall-clock time of ``session.run`` in milliseconds.
        """
        self._ensure_session()
        t0 = time.perf_counter()
        outputs = self._session.run(
            self._output_names, {self._input_name: tensor}
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return outputs, elapsed_ms

    # -- Preprocessing helper -----------------------------------------------

    def _preprocess_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Resize and normalise an RGB image to the model's NCHW input tensor.

        Used by U-Net (direct resize without letterboxing).  YOLOv8 overrides
        this step with aspect-ratio-preserving letterboxing handled inside its
        own ``predict`` implementation.

        Parameters
        ----------
        image_rgb:
            RGB uint8 array of arbitrary shape (H, W, 3) or (H, W) for
            single-channel models.

        Returns
        -------
        numpy.ndarray
            Float32 NCHW tensor of shape ``(1, C, H, W)`` ready for
            ``_run``.
        """
        import cv2

        h, w = self.spec.input_size

        if self.spec.channels == 1:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
            arr = resized.astype(np.float32)[None, ...]      # (1, H, W)
        else:
            resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            arr = resized.astype(np.float32).transpose(2, 0, 1)  # (3, H, W)

        if self.spec.normalize == "standard":
            mean = np.array(self.spec.mean, dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(self.spec.std, dtype=np.float32).reshape(-1, 1, 1)
            arr = (arr / 255.0 - mean) / std
        else:  # "scale" — divide by 255 only
            arr = arr / 255.0

        return arr[None, ...].astype(np.float32)  # (1, C, H, W)

    # -- Abstract contract --------------------------------------------------

    @abstractmethod
    def predict(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Segment the fetal skull in a uint8 RGB image.

        Parameters
        ----------
        image_rgb:
            Input image as a uint8 RGB array of shape (H, W, 3).

        Returns
        -------
        SegmentationResult
            Binary skull mask at the original resolution, timing, and
            confidence information.
        """
        raise NotImplementedError
