"""
registry.py
===========
Factory and availability helpers for the configured segmentation models.

Keeps Streamlit out of the inference layer: the application wraps
``build_segmenter`` in ``st.cache_resource`` so that each ONNX Runtime
session is created once per server process, regardless of the number of
connected browser sessions.

Responsibilities
----------------
* ``build_segmenter`` — instantiate the correct ``Segmenter`` subclass for a
  given model key (does not load weights; the ONNX session is initialised
  lazily on first ``predict`` call).
* ``model_status`` / ``all_statuses`` — report weight-file availability for
  the UI availability chips and model selector.
* ``any_available`` — quick gate to determine whether the application can
  perform inference at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config
from config import ModelSpec
from .base import Segmenter
from .yolov8_onnx import YOLOv8Segmenter
from .unet_onnx import UNetSegmenter

# Mapping from ModelSpec.family identifier to the corresponding Segmenter class.
_FAMILIES: dict[str, type[Segmenter]] = {
    "yolov8_seg": YOLOv8Segmenter,
    "unet": UNetSegmenter,
}


@dataclass(frozen=True)
class ModelStatus:
    """Snapshot of a model's configuration and runtime availability.

    Attributes
    ----------
    key:
        Internal registry key matching ``ModelSpec.key``.
    display_name:
        Human-readable name shown in the UI toggle.
    family:
        Architecture family identifier (``"yolov8_seg"`` or ``"unet"``).
    available:
        ``True`` when the ONNX weights file exists on disk and inference can
        proceed.
    weights_path:
        Absolute path to the expected ``.onnx`` file, as a string.
    description:
        Short model description shown below the selector in the UI.
    """

    key: str
    display_name: str
    family: str
    available: bool
    weights_path: str
    description: str


def build_segmenter(model_key: str) -> Segmenter:
    """Instantiate the ``Segmenter`` for ``model_key`` without loading weights.

    The ONNX Runtime session is created lazily on the first call to
    ``predict``; this function only resolves the correct class and wires it
    to the ``ModelSpec``.

    Parameters
    ----------
    model_key:
        Registry key identifying the model, e.g. ``"yolov8"`` or ``"unet"``.

    Returns
    -------
    Segmenter
        Un-initialised segmenter instance ready for deferred session creation.

    Raises
    ------
    KeyError
        If ``model_key`` is not present in ``config.MODELS``.
    ValueError
        If the model's ``family`` field does not map to a known implementation.
    """
    spec: ModelSpec = config.MODELS[model_key]
    cls = _FAMILIES.get(spec.family)
    if cls is None:
        raise ValueError(
            f"Unknown model family '{spec.family}' for key '{model_key}'. "
            f"Supported families: {list(_FAMILIES.keys())}."
        )
    return cls(spec)


def model_status(model_key: str) -> ModelStatus:
    """Return the configuration and availability status for one model.

    Parameters
    ----------
    model_key:
        Registry key identifying the model.

    Returns
    -------
    ModelStatus
        Snapshot suitable for driving UI availability indicators.
    """
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
    """Return the status of every configured model, in UI display order.

    The order follows ``config.MODEL_ORDER``.

    Returns
    -------
    list[ModelStatus]
        One entry per model registered in ``config.MODELS``.
    """
    return [model_status(k) for k in config.MODEL_ORDER]


def any_available() -> bool:
    """Return ``True`` if at least one model's weights file is present on disk."""
    return any(s.available for s in all_statuses())
