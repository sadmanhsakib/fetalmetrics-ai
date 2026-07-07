"""
yolov8_onnx.py
==============
YOLOv8-segmentation inference from an exported ``.onnx`` model.

Expected export (Ultralytics default for a segmentation model)
--------------------------------------------------------------
* input  : (1, 3, H, W) RGB, normalized /255, letterboxed.
* output0: (1, 4 + nc + 32, num_anchors) detection+coefficient head.
* output1: (1, 32, mh, mw) mask prototypes.

Number of classes ``nc`` is inferred from the head width, so single-class
"fetal head" exports work without changes. If your export differs, adjust
``ModelSpec`` in config.py (input size, normalization).

Post-processing takes the highest-confidence detection (the skull), builds its
mask from the prototype basis, removes the letterbox padding and resizes back
to the original resolution — yielding a mask the shared ellipse/HC pipeline can
consume directly.
"""

from __future__ import annotations

import numpy as np

import config
from .base import Segmenter, SegmentationResult, letterbox, sigmoid


class YOLOv8Segmenter(Segmenter):
    """YOLOv8-seg ONNX wrapper."""

    NUM_MASK_COEFFS = 32

    def predict(self, image_rgb: np.ndarray) -> SegmentationResult:
        import cv2

        h0, w0 = image_rgb.shape[:2]
        in_h, in_w = self.spec.input_size

        # -- preprocess (letterbox) --------------------------------------- #
        padded, ratio, (pad_left, pad_top) = letterbox(image_rgb, (in_h, in_w))
        tensor = padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)

        outputs, elapsed_ms = self._run(tensor)

        empty = np.zeros((h0, w0), dtype=np.uint8)
        if len(outputs) < 2:
            # Not a segmentation export — cannot build a mask.
            return SegmentationResult(empty, elapsed_ms, self.name, found=False)

        preds, protos = outputs[0], outputs[1]

        # -- normalize head layout to (num_anchors, features) ------------- #
        preds = np.asarray(preds)
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[1]:      # (features, anchors) -> transpose
            preds = preds.T
        num_features = preds.shape[1]
        nc = num_features - 4 - self.NUM_MASK_COEFFS
        if nc < 1:
            return SegmentationResult(empty, elapsed_ms, self.name, found=False)

        boxes = preds[:, :4]
        cls_scores = preds[:, 4:4 + nc]
        coeffs = preds[:, 4 + nc:]
        conf = cls_scores.max(axis=1)
        if len(conf) == 0:
            return SegmentationResult(empty, elapsed_ms, self.name, found=False, confidence=0.0)

        keep = conf >= self.spec.conf_threshold
        if not np.any(keep):
            return SegmentationResult(empty, elapsed_ms, self.name,
                                       found=False, confidence=float(conf.max()))

        boxes, coeffs, conf = boxes[keep], coeffs[keep], conf[keep]

        # Highest-confidence detection = the skull.
        best = int(np.argmax(conf))
        best_conf = float(conf[best])
        cx, cy, bw, bh = boxes[best]

        # -- assemble mask from prototype basis --------------------------- #
        protos = np.asarray(protos)[0]                     # (32, mh, mw)
        ch, mh, mw = protos.shape
        mask_small = sigmoid(coeffs[best] @ protos.reshape(ch, -1)).reshape(mh, mw)
        mask_pad = cv2.resize(mask_small, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

        # Restrict to the detection box (in letterboxed coords).
        x1 = int(max(0, cx - bw / 2)); y1 = int(max(0, cy - bh / 2))
        x2 = int(min(in_w, cx + bw / 2)); y2 = int(min(in_h, cy + bh / 2))
        box_mask = np.zeros_like(mask_pad)
        box_mask[y1:y2, x1:x2] = mask_pad[y1:y2, x1:x2]
        binary_pad = (box_mask >= config.POSTPROCESS["mask_threshold"]).astype(np.uint8)

        # -- remove letterbox padding, resize to original ----------------- #
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
