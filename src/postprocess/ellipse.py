"""
ellipse.py
==========
Turn a binary skull mask into a calibrated head-circumference measurement.

Pipeline
--------
1. Clean the mask (morphological closing) and take the largest plausible
   contour — the fetal skull.
2. Fit an ellipse to that contour (``cv2.fitEllipse``, direct least-squares).
3. Convert the ellipse's semi-axes from pixels to millimetres using the image
   calibration.
4. Compute the perimeter (= head circumference) with the **Ramanujan II**
   approximation, matching the HC18 ground-truth convention.

``ramanujan_perimeter`` is pure math and importable without OpenCV, so the
clinical/geometry maths can be unit-tested in isolation. OpenCV is imported
lazily only inside the functions that need it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config


# --------------------------------------------------------------------------- #
# Pure geometry (no OpenCV)
# --------------------------------------------------------------------------- #
def ramanujan_perimeter(semi_major: float, semi_minor: float) -> float:
    """Ellipse perimeter via Ramanujan's second approximation.

    ``C ≈ π (a + b) [ 1 + 3h / (10 + sqrt(4 - 3h)) ]`` with
    ``h = ((a - b) / (a + b))**2``. Accurate to a few ppm for realistic fetal
    skull eccentricities.

    Parameters
    ----------
    semi_major, semi_minor:
        Semi-axes in any consistent unit; the result is in that same unit.
    """
    a = float(semi_major)
    b = float(semi_minor)
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(4.0 - 3.0 * h)))


@dataclass(frozen=True)
class HCResult:
    """A calibrated head-circumference measurement and its geometry."""

    hc_mm: float
    hc_px: float
    center_px: tuple[float, float]
    major_axis_px: float          # full major-axis length (px)
    minor_axis_px: float          # full minor-axis length (px)
    angle_deg: float              # orientation of the major axis
    major_axis_mm: float
    minor_axis_mm: float
    mm_per_px: float
    contour_area_px: float
    # Exact cv2.fitEllipse output ((cx, cy), (axis_a, axis_b), angle) kept so the
    # overlay draws the true fitted ellipse and an axis-aligned crosshair.
    ellipse_cv: tuple[tuple[float, float], tuple[float, float], float] = (
        (0.0, 0.0), (0.0, 0.0), 0.0
    )


# --------------------------------------------------------------------------- #
# OpenCV-backed steps
# --------------------------------------------------------------------------- #
def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Binarize + morphologically close a raw mask so contours are solid."""
    import cv2

    binary = (mask > 0).astype(np.uint8) * 255
    k = int(config.POSTPROCESS["morph_kernel"])
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def largest_contour(mask: np.ndarray):
    """Return the largest external contour meeting the min-area threshold.

    Returns ``None`` when no contour is large enough to be a skull.
    """
    import cv2

    binary = _clean_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = float(mask.shape[0] * mask.shape[1])
    min_area = config.POSTPROCESS["min_area_frac"] * img_area
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < min_area or len(biggest) < 5:
        # fitEllipse needs >= 5 points.
        return None
    return biggest


def measure_hc(mask: np.ndarray, mm_per_px: float) -> HCResult | None:
    """Fit an ellipse to the skull mask and return the calibrated HC.

    Parameters
    ----------
    mask:
        2-D array; non-zero pixels are the predicted fetal skull.
    mm_per_px:
        Physical scale (millimetres per pixel) from calibration.

    Returns
    -------
    HCResult | None
        ``None`` if no ellipse could be fitted.
    """
    import cv2

    contour = largest_contour(mask)
    if contour is None:
        return None

    try:
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    except cv2.error:
        return None

    # cv2 does not guarantee ordering; enforce major >= minor.
    major_px, minor_px = (axis_a, axis_b) if axis_a >= axis_b else (axis_b, axis_a)

    if major_px <= 0 or minor_px <= 0:
        return None

    semi_major_mm = (major_px / 2.0) * mm_per_px
    semi_minor_mm = (minor_px / 2.0) * mm_per_px

    hc_mm = ramanujan_perimeter(semi_major_mm, semi_minor_mm)
    hc_px = ramanujan_perimeter(major_px / 2.0, minor_px / 2.0)

    return HCResult(
        hc_mm=hc_mm,
        hc_px=hc_px,
        center_px=(float(cx), float(cy)),
        major_axis_px=float(major_px),
        minor_axis_px=float(minor_px),
        angle_deg=float(angle),
        major_axis_mm=semi_major_mm * 2.0,
        minor_axis_mm=semi_minor_mm * 2.0,
        mm_per_px=float(mm_per_px),
        contour_area_px=float(cv2.contourArea(contour)),
        ellipse_cv=((float(cx), float(cy)), (float(axis_a), float(axis_b)), float(angle)),
    )
