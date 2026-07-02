"""
unet_onnx.py
============
U-Net semantic-segmentation inference from an exported ``.onnx`` model.

Accepted output layouts (auto-detected)
---------------------------------------
* (1, 1, H, W) or (1, H, W) — single-channel logits or probabilities.
* (1, 2, H, W)              — two-class logits (background, skull); channel 1
  is taken after softmax.

If values fall outside [0, 1] a sigmoid/softmax is applied (the export produced
logits); otherwise they are treated as probabilities. The probability map is
thresholded and resized back to the original resolution.

Input channels and normalization are driven by ``ModelSpec`` in config.py, so a
grayscale 1-channel U-Net and a 3-channel variant are both supported.
"""

from __future__ import annotations

import numpy as np

import config
from .base import Segmenter, SegmentationResult, sigmoid


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class UNetSegmenter(Segmenter):
    """U-Net ONNX wrapper."""

    def predict(self, image_rgb: np.ndarray) -> SegmentationResult:
        import cv2

        h0, w0 = image_rgb.shape[:2]
        tensor = self._preprocess_image(image_rgb)
        outputs, elapsed_ms = self._run(tensor)

        out = np.asarray(outputs[0])

        # Collapse to a single-channel probability map at network resolution.
        if out.ndim == 4 and out.shape[1] == 2:            # (1, 2, H, W)
            prob = _softmax(out, axis=1)[0, 1]
        elif out.ndim == 4:                                # (1, 1, H, W)
            prob = out[0, 0]
        elif out.ndim == 3:                                # (1, H, W)
            prob = out[0]
        else:                                              # (H, W)
            prob = out

        if prob.min() < 0.0 or prob.max() > 1.0:           # logits -> probabilities
            prob = sigmoid(prob)

        binary = (prob >= config.POSTPROCESS["mask_threshold"]).astype(np.uint8)
        mask = cv2.resize(binary, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # Confidence proxy: mean probability over the predicted region.
        conf = float(prob[prob >= config.POSTPROCESS["mask_threshold"]].mean()) \
            if binary.any() else 0.0

        return SegmentationResult(
            mask=mask,
            inference_ms=elapsed_ms,
            model_name=self.name,
            found=bool(mask.any()),
            confidence=conf,
        )
