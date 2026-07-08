"""
unet_onnx.py
============
U-Net semantic-segmentation inference from an exported ``.onnx`` model.

Accepted output layouts (auto-detected)
----------------------------------------
* ``(1, 1, H, W)`` or ``(1, H, W)`` — single-channel logits or probabilities.
* ``(1, 2, H, W)``                  — two-class logits (background, fetal head);
  the foreground channel (index 1) is extracted after softmax normalisation.

Logit detection
    When the output values fall outside [0, 1], a sigmoid (single-channel) or
    softmax (two-channel) is applied automatically.  Values already in [0, 1]
    are treated as probabilities and passed through unchanged.

The resulting probability map is thresholded at ``config.POSTPROCESS["mask_threshold"]``
and nearest-neighbour-resized back to the original image resolution, yielding
a binary mask compatible with the shared ``ellipse`` / HC pipeline.

Input channel count and normalisation are driven by ``ModelSpec`` in
``config.py``, so a grayscale 1-channel U-Net and a 3-channel RGB variant
are both supported without code changes.
"""

from __future__ import annotations

import numpy as np

import config
from .base import Segmenter, SegmentationResult, sigmoid


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable softmax along ``axis``.

    Subtracts the per-sample maximum before exponentiation to prevent
    overflow without changing the output distribution.

    Parameters
    ----------
    x:
        Input array of arbitrary shape.
    axis:
        Axis along which to normalise.

    Returns
    -------
    numpy.ndarray
        Softmax-normalised array with the same shape as ``x``.
    """
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class UNetSegmenter(Segmenter):
    """ONNX Runtime wrapper for a U-Net semantic-segmentation model.

    Inherits session management and image preprocessing from ``Segmenter``.
    Only the output-decoding logic is specific to U-Net and is implemented
    here in ``predict``.
    """

    def predict(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Segment the fetal skull in an RGB uint8 image.

        The method handles output-layout detection automatically, converting
        logits to probabilities where necessary before thresholding.

        Parameters
        ----------
        image_rgb:
            Input image as a uint8 RGB array of shape (H, W, 3).

        Returns
        -------
        SegmentationResult
            Binary skull mask at the original resolution, inference timing,
            and a confidence proxy (mean probability over the foreground).
        """
        import cv2

        h0, w0 = image_rgb.shape[:2]
        tensor = self._preprocess_image(image_rgb)
        outputs, elapsed_ms = self._run(tensor)

        out = np.asarray(outputs[0])

        # Collapse the model output to a single-channel probability map.
        if out.ndim == 4 and out.shape[1] == 2:    # (1, 2, H, W) two-class logits
            prob = _softmax(out, axis=1)[0, 1]
        elif out.ndim == 4:                          # (1, 1, H, W) single-channel
            prob = out[0, 0]
        elif out.ndim == 3:                          # (1, H, W) squeezed batch
            prob = out[0]
        else:                                        # (H, W) fully squeezed
            prob = out

        # Apply sigmoid if the values indicate the output is in logit space.
        if prob.min() < 0.0 or prob.max() > 1.0:
            prob = sigmoid(prob)

        binary = (prob >= config.POSTPROCESS["mask_threshold"]).astype(np.uint8)
        mask = cv2.resize(
            binary, (w0, h0), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

        # Confidence proxy: mean predicted probability over the foreground
        # region.  Returns 0.0 when the mask is entirely empty.
        foreground = prob[prob >= config.POSTPROCESS["mask_threshold"]]
        conf = float(foreground.mean()) if binary.any() else 0.0

        return SegmentationResult(
            mask=mask,
            inference_ms=elapsed_ms,
            model_name=self.name,
            found=bool(mask.any()),
            confidence=conf,
        )
