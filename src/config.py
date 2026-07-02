"""
config.py
=========
Central, auditable configuration for fetalmetrics-ai.

Every tunable that affects a clinical or geometric result lives here so that it
can be reviewed in one place. Values that must match your exported ONNX models
(input size, normalization, tensor layout) are grouped under ``MODELS`` and are
intended to be adjusted to your training/export pipeline.

Nothing here performs I/O or heavy imports, so it is safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

ASSETS_DIR: Path = ROOT_DIR / "src" / "assets"
DATA_DIR: Path = ROOT_DIR / "data" / "raw"
MODELS_DIR: Path = ROOT_DIR / "models"
# HC18 pixel-size lookup tables (filename -> mm/pixel). Either file may be
# present; both are searched. See src/calibration/pixel_size.py.
HC18_PIXEL_SIZE_CSVS: tuple[Path, ...] = (
    DATA_DIR / "training_set_pixel_size_and_HC.csv",
    DATA_DIR / "test_set_pixel_size.csv",
)

# --------------------------------------------------------------------------- #
# Application metadata
# --------------------------------------------------------------------------- #
APP_NAME: str = "Fetal Metrics-ai"
APP_TAGLINE: str = "Automated Fetal Biometry Assistant"
APP_VERSION: str = "1.0.0"
SAFETY_NOTICE: str = (
    "This is a research prototype and is not a certified diagnostic device. "
    "Outputs are for engineering demonstration and research only and must not "
    "be used for clinical decision-making."
)

# --------------------------------------------------------------------------- #
# Gestational age input constraints (drives the slider + reference lookups)
# --------------------------------------------------------------------------- #
GA_MIN_WEEKS: float = 14.0
GA_MAX_WEEKS: float = 40.0
GA_STEP_WEEKS: float = 0.5
GA_DEFAULT_WEEKS: float = 28.0

# --------------------------------------------------------------------------- #
# Clinical risk bands (percentile thresholds) + semantic colors
# Colors are used ONLY to encode risk, never for decoration.
# --------------------------------------------------------------------------- #
HIGH_RISK_MAX_PCT: float = 10.0     # percentile < 10  -> High risk (IUGR alert)
MEDIUM_RISK_MAX_PCT: float = 25.0   # 10 <= pct < 25    -> Medium (borderline)
# pct >= 25 -> Normal

RISK_COLORS: dict[str, dict[str, str]] = {
    "HIGH": {"solid": "#EF4444", "accent": "#F43F5E", "soft": "rgba(239,68,68,0.14)"},
    "MEDIUM": {"solid": "#F59E0B", "accent": "#D97706", "soft": "rgba(245,158,11,0.14)"},
    "NORMAL": {"solid": "#10B981", "accent": "#059669", "soft": "rgba(16,185,129,0.14)"},
}

# --------------------------------------------------------------------------- #
# Overlay rendering
# --------------------------------------------------------------------------- #
# Colors are given as RGB (matplotlib/PIL convention). The overlay module
# converts to BGR internally when it hands off to OpenCV.
OVERLAY = {
    "ellipse_rgb": (0, 240, 255),      # electric cyan  (#00F0FF)
    "crosshair_rgb": (245, 158, 11),   # amber axes for contrast on cyan/gray
    "mask_fill_rgb": (0, 240, 255),
    "mask_fill_alpha": 0.16,
    "ellipse_thickness": 2,
    "crosshair_thickness": 1,
}

# --------------------------------------------------------------------------- #
# Segmentation post-processing
# --------------------------------------------------------------------------- #
POSTPROCESS = {
    # Probability threshold applied to model output to obtain a binary mask.
    "mask_threshold": 0.5,
    # Morphological closing kernel (px) to seal small gaps before contouring.
    "morph_kernel": 5,
    # Minimum contour area (fraction of image) to be considered the fetal skull.
    "min_area_frac": 0.01,
}


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Everything the inference layer needs to run one exported ONNX model.

    Adjust ``input_size``, ``normalize``, ``mean``/``std`` and the family flag
    to match how *you* exported the network. These are the only values you
    should ever need to touch to wire in your own weights.
    """

    key: str                       # internal id, e.g. "yolov8"
    display_name: str              # shown in the UI toggle
    family: str                    # "yolov8_seg" | "unet"
    weights_path: Path             # .onnx file location
    input_size: tuple[int, int]    # (height, width) fed to the network
    channels: int = 3              # 1 = grayscale, 3 = RGB
    normalize: str = "scale"       # "scale" (/255) or "standard" (mean/std)
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
        description="Single-stage detector+prototype-mask head. Fast inference.",
    ),
    "unet": ModelSpec(
        key="unet",
        display_name="U-Net",
        family="unet",
        weights_path=MODELS_DIR / "unet_hc.onnx",
        input_size=(512, 512),
        channels=1,
        normalize="scale",
        description="Encoder-decoder semantic segmentation. Dense pixel mask.",
    ),
}

# Order of models in the UI toggle. First entry is the default selection.
MODEL_ORDER: tuple[str, ...] = ("yolov8", "unet")

# Default manual calibration used only when no HC18 metadata is available and
# the user has not entered a value. Typical HC18 images fall near this range.
DEFAULT_PIXEL_SIZE_MM: float = 0.15
