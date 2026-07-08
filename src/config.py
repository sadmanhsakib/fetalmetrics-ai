"""
config.py
=========
Centralised, auditable configuration for FetalMetrics-AI.

Every parameter that affects a clinical or geometric result lives here so that
it can be reviewed and modified in a single location.  Parameters that must
match your exported ONNX models (input size, normalization, tensor layout) are
grouped under ``MODELS`` and are designed to be adjusted to match your own
training and export pipeline.

Design constraints
------------------
* No I/O or heavy imports — safe to import in any context, including unit
  tests that do not have access to model weights or datasets.
* All values are typed and annotated for IDE-assisted review.
* The module is imported by both the runtime inference path and the generated
  Methodology documentation page, so editing a value here propagates
  throughout the entire application automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parent.parent

ASSETS_DIR: Path = ROOT_DIR / "src" / "assets"
DATA_DIR: Path = ROOT_DIR / "data" / "raw"
MODELS_DIR: Path = ROOT_DIR / "models"

# HC18 pixel-size look-up tables mapping image filename to mm/pixel.  Either
# file may be present independently; both are searched on import.
# See ``src/calibration/pixel_size.py`` for the resolution logic.
HC18_PIXEL_SIZE_CSVS: tuple[Path, ...] = (
    DATA_DIR / "training_set_pixel_size_and_HC.csv",
    DATA_DIR / "test_set_pixel_size.csv",
)

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

APP_NAME: str = "FetalMetrics-AI"
APP_TAGLINE: str = "Automated Fetal Biometry Assistant"
SAFETY_NOTICE: str = (
    "This tool is intended for research and educational purposes only. "
    "It is NOT a diagnostic medical device and should NOT be used as the sole "
    "basis for clinical decisions. All measurements must be verified by "
    "qualified healthcare professionals."
)

# ---------------------------------------------------------------------------
# Gestational age slider constraints
# Drives both the UI slider range and reference-curve interpolation limits.
# ---------------------------------------------------------------------------

GA_MIN_WEEKS: float = 14.0
GA_MAX_WEEKS: float = 40.0
GA_STEP_WEEKS: float = 0.5
GA_DEFAULT_WEEKS: float = 28.0

# ---------------------------------------------------------------------------
# Clinical risk bands (percentile thresholds)
#
# Colors encode risk semantics only — never used for decoration.  All six
# token values must remain in sync with the ``--risk-*`` CSS custom properties
# defined in ``assets/styles.css``.  The hue choices are calibrated for
# maximum legibility on the clinical-light canvas at -600 weight.
# ---------------------------------------------------------------------------

HIGH_RISK_MAX_PCT: float = 10.0    # percentile < 10  → High Risk  (IUGR alert)
MEDIUM_RISK_MAX_PCT: float = 25.0  # 10 ≤ pct < 25   → Medium Risk (borderline)
# percentile ≥ 25  → Normal

RISK_COLORS: dict[str, dict[str, str]] = {
    "HIGH": {
        "solid": "#DC2626",
        "accent": "#B91C1C",
        "soft": "rgba(220,38,38,0.08)",
    },
    "MEDIUM": {
        "solid": "#D97706",
        "accent": "#B45309",
        "soft": "rgba(217,119,6,0.09)",
    },
    "NORMAL": {
        "solid": "#16A34A",
        "accent": "#15803D",
        "soft": "rgba(22,163,74,0.09)",
    },
}

# ---------------------------------------------------------------------------
# Overlay rendering
#
# Colors are given in RGB order (matplotlib / PIL convention).  The overlay
# module converts to BGR internally when handing off to OpenCV drawing calls.
# ---------------------------------------------------------------------------

OVERLAY: dict = {
    "ellipse_rgb": (0, 240, 255),      # Electric cyan (#00F0FF) — skull boundary
    "crosshair_rgb": (245, 158, 11),   # Warm amber — axis lines, readable on cyan
    "mask_fill_rgb": (0, 240, 255),    # Matches ellipse color for visual coherence
    "mask_fill_alpha": 0.16,           # Subtle translucent tint; avoids obscuring anatomy
    "ellipse_thickness": 2,            # Pixels; thin enough to not hide the boundary
    "crosshair_thickness": 1,          # Thinner than ellipse to maintain hierarchy
}

# ---------------------------------------------------------------------------
# Segmentation post-processing
# ---------------------------------------------------------------------------

POSTPROCESS: dict = {
    # Probability threshold applied to the model's output to produce a binary
    # foreground mask.
    "mask_threshold": 0.5,
    # Morphological closing kernel diameter (pixels) used to seal small
    # segmentation gaps before contour extraction.
    "morph_kernel": 5,
    # Minimum acceptable contour area expressed as a fraction of total image
    # area.  Contours below this size are treated as noise and discarded.
    "min_area_frac": 0.01,
}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Complete specification for one exported ONNX segmentation model.

    Adjust ``input_size``, ``normalize``, ``mean``, ``std``, and
    ``conf_threshold`` / ``iou_threshold`` to match your own training and
    export pipeline.  These are the only values that need to change when
    swapping in custom weights.

    Attributes
    ----------
    key:
        Internal registry key, e.g. ``"yolov8"``.
    display_name:
        Short label shown in the UI model toggle.
    family:
        Architecture family selector.  Must be one of ``"yolov8_seg"`` or
        ``"unet"``.
    weights_path:
        Absolute path to the ``.onnx`` file.
    input_size:
        Spatial dimensions fed to the network as ``(height, width)``.
    channels:
        Number of input channels: 1 for grayscale, 3 for RGB.
    normalize:
        Normalization strategy.  ``"scale"`` divides by 255; ``"standard"``
        applies per-channel mean/std subtraction after dividing by 255.
    mean:
        Per-channel mean used when ``normalize="standard"``.
    std:
        Per-channel standard deviation used when ``normalize="standard"``.
    conf_threshold:
        Minimum detection confidence required to accept a YOLOv8 prediction.
        Ignored by U-Net.
    iou_threshold:
        IoU threshold for non-maximum suppression in YOLOv8.  Ignored by
        U-Net.
    description:
        Short human-readable description shown below the model selector in
        the UI.
    """

    key: str
    display_name: str
    family: str                         # "yolov8_seg" | "unet"
    weights_path: Path
    input_size: tuple[int, int]         # (height, width) fed to the network
    channels: int = 3                   # 1 = grayscale, 3 = RGB
    normalize: str = "scale"            # "scale" (/255) or "standard" (mean/std)
    mean: tuple[float, ...] = (0.0,)
    std: tuple[float, ...] = (1.0,)
    # YOLOv8-seg specifics (ignored for U-Net):
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    description: str = ""


MODELS: dict[str, ModelSpec] = {
    "yolov8": ModelSpec(
        key="yolov8",
        display_name="YOLOv8-seg",
        family="yolov8_seg",
        weights_path=MODELS_DIR / "yolov8_hc.onnx",
        input_size=(640, 640),
        channels=3,
        normalize="scale",
        conf_threshold=0.25,
        iou_threshold=0.45,
        description="Single-stage detector + prototype-mask head. Fast inference.",
    ),
    "unet": ModelSpec(
        key="unet",
        display_name="U-Net",
        family="unet",
        weights_path=MODELS_DIR / "unet_hc.onnx",
        input_size=(256, 256),
        channels=3,
        normalize="standard",
        mean=(0.485, 0.456, 0.406),   # ImageNet channel means — must match training
        std=(0.229, 0.224, 0.225),    # ImageNet channel stds  — must match training
        description="Encoder-decoder semantic segmentation. Dense pixel mask.",
    ),
}

# Ordering of models in the UI toggle.  The first entry is the default
# selection on page load.
MODEL_ORDER: tuple[str, ...] = ("yolov8", "unet")

# Default pixel scale applied only when no HC18 metadata is available and the
# user has not entered a value.  Typical HC18 images fall near this range.
DEFAULT_PIXEL_SIZE_MM: float = 0.15
