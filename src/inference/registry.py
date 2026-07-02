"""
registry.py
===========
Factory + availability helpers for the segmentation models.

Keeps Streamlit out of the inference layer: the app wraps ``build_segmenter``
in ``st.cache_resource`` so each ONNX session is created once per process.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config
from config import ModelSpec
from .base import Segmenter
from .yolov8_onnx import YOLOv8Segmenter
from .unet_onnx import UNetSegmenter

_FAMILIES: dict[str, type[Segmenter]] = {
    "yolov8_seg": YOLOv8Segmenter,
    "unet": UNetSegmenter,
}


@dataclass(frozen=True)
class ModelStatus:
    key: str
    display_name: str
    family: str
    available: bool
    weights_path: str
    description: str


def build_segmenter(model_key: str) -> Segmenter:
    """Instantiate the segmenter for ``model_key`` (does not load weights yet)."""
    spec: ModelSpec = config.MODELS[model_key]
    cls = _FAMILIES.get(spec.family)
    if cls is None:
        raise ValueError(f"Unknown model family '{spec.family}' for '{model_key}'.")
    return cls(spec)


def model_status(model_key: str) -> ModelStatus:
    """Report whether a model's weights are present, for UI display."""
    spec = config.MODELS[model_key]
    return ModelStatus(
        key=spec.key,
        display_name=spec.display_name,
        family=spec.family,
        available=Path(spec.weights_path).exists(),
        weights_path=str(spec.weights_path),
        description=spec.description,
    )


def all_statuses() -> list[ModelStatus]:
    """Status for every configured model, in UI order."""
    return [model_status(k) for k in config.MODEL_ORDER]


def any_available() -> bool:
    return any(s.available for s in all_statuses())
