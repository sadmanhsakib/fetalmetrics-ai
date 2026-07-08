"""
yolov8_onnx.py
==============
YOLOv8 instance-segmentation inference from an exported ``.onnx`` model.

Expected ONNX export layout (Ultralytics default for a segmentation model)
---------------------------------------------------------------------------
* ``input``   : ``(1, 3, H, W)`` — RGB, normalised to [0, 1], letterboxed.
* ``output0`` : ``(1, 4 + nc + 32, num_anchors)`` — detection and coefficient
  head.  ``nc`` is the number of classes; for a single-class "fetal head"
  model, ``nc = 1``.
* ``output1`` : ``(1, 32, mh, mw)`` — prototype mask tensor.

Post-processing pipeline
------------------------
1. Normalise the detection head to ``(num_anchors, features)``.
2. Extract class scores and select all anchors whose maximum class score
   exceeds ``ModelSpec.conf_threshold``.
3. Pick the highest-confidence detection (the skull is the dominant object).
4. Reconstruct the instance mask as a linear combination of the 32 prototype
   masks weighted by the detection's 32 mask coefficients.
5. Threshold, crop to the detection bounding box, strip letterbox padding,
   and resize to the original image resolution.

If the model export differs (e.g. different input size or normalization),
adjust ``ModelSpec`` in ``config.py`` — no code changes are required here.
"""

from __future__ import annotations

import numpy as np

import config
from .base import Segmenter, SegmentationResult, letterbox, sigmoid


class YOLOv8Segmenter(Segmenter):
    """ONNX Runtime wrapper for a YOLOv8-seg instance-segmentation model.

    Inherits session management from ``Segmenter`` and provides a complete
    YOLOv8 post-processing pipeline inside ``predict``.
    """

    NUM_MASK_COEFFS: int = 32  # Prototype mask count — fixed by the YOLOv8-seg architecture.

    def predict(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Segment the fetal skull in an RGB uint8 image using YOLOv8-seg.

        Parameters
        ----------
        image_rgb:
            Input image as a uint8 RGB array of shape (H, W, 3).

        Returns
        -------
        SegmentationResult
            Binary skull mask at the original resolution, inference timing,
            and detection confidence.  Returns an empty mask with
            ``found=False`` when no anchor exceeds the confidence threshold or
            when the ONNX export does not include a segmentation head.
        """
        import cv2

        h0, w0 = image_rgb.shape[:2]
        in_h, in_w = self.spec.input_size

        # -- Pre-processing (letterbox) -------------------------------------
        # Preserve aspect ratio by uniform scaling then padding, consistent
        # with Ultralytics' standard export pre-processing transform.
        padded, ratio, (pad_left, pad_top) = letterbox(image_rgb, (in_h, in_w))
        tensor = padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)

        outputs, elapsed_ms = self._run(tensor)

        empty = np.zeros((h0, w0), dtype=np.uint8)

        if len(outputs) < 2:
            # The ONNX export does not include a segmentation head (output1 is
            # missing); a detection-only export cannot produce a mask.
            return SegmentationResult(empty, elapsed_ms, self.name, found=False)

        preds, protos = outputs[0], outputs[1]

        # -- Normalise the detection head layout to (num_anchors, features) ---
        # The Ultralytics exporter may produce either (1, features, anchors)
        # or (1, anchors, features); both are handled here.
        preds = np.asarray(preds)
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[1]:   # (features, anchors) → transpose
            preds = preds.T

        num_features = preds.shape[1]
        nc = num_features - 4 - self.NUM_MASK_COEFFS

        if nc < 1:
            # Cannot determine the class count — the export layout is unexpected.
            return SegmentationResult(empty, elapsed_ms, self.name, found=False)

        boxes = preds[:, :4]
        cls_scores = preds[:, 4:4 + nc]
        coeffs = preds[:, 4 + nc:]
        conf = cls_scores.max(axis=1)

        if len(conf) == 0:
            return SegmentationResult(empty, elapsed_ms, self.name, found=False, confidence=0.0)

        # Filter anchors below the configured confidence threshold.
        keep = conf >= self.spec.conf_threshold
        if not np.any(keep):
            return SegmentationResult(
                empty, elapsed_ms, self.name, found=False, confidence=float(conf.max())
            )

        boxes, coeffs, conf = boxes[keep], coeffs[keep], conf[keep]

        # Select the highest-confidence detection as the skull candidate.
        best = int(np.argmax(conf))
        best_conf = float(conf[best])
        cx, cy, bw, bh = boxes[best]

        # -- Assemble the instance mask from prototype basis ----------------
        # mask_small = sigmoid(coefficients · prototypes)
        protos = np.asarray(protos)[0]          # (32, mh, mw)
        ch, mh, mw = protos.shape
        mask_small = sigmoid(
            coeffs[best] @ protos.reshape(ch, -1)
        ).reshape(mh, mw)

        # Upscale from prototype resolution to the letterboxed input resolution.
        mask_pad = cv2.resize(mask_small, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

        # Crop mask to the detection bounding box (in letterboxed coordinates)
        # to suppress background activations outside the detected region.
        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(in_w, cx + bw / 2))
        y2 = int(min(in_h, cy + bh / 2))
        box_mask = np.zeros_like(mask_pad)
        box_mask[y1:y2, x1:x2] = mask_pad[y1:y2, x1:x2]

        binary_pad = (box_mask >= config.POSTPROCESS["mask_threshold"]).astype(np.uint8)

        # -- Strip letterbox padding and resize to original resolution -------
        nh, nw = int(round(h0 * ratio)), int(round(w0 * ratio))
        crop = binary_pad[pad_top:pad_top + nh, pad_left:pad_left + nw]
        mask = cv2.resize(crop, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        return SegmentationResult(
            mask=mask,
            inference_ms=elapsed_ms,
            model_name=self.name,
            found=bool(mask.any()),
            confidence=best_conf,
        )
