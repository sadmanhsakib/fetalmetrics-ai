"""Module for geometric ellipse fitting and circumference extraction."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class EllipseParams:
    """
    Dataclass holding fitted ellipse geometry and physical measurements.

    Attributes:
        center_x: Center X coordinate in pixels.
        center_y: Center Y coordinate in pixels.
        axes_width_px: Full diameter along the width axis in pixels (NOT the semi-axis radius).
        axes_height_px: Full diameter along the height axis in pixels (NOT the semi-axis radius).
        angle_degrees: Rotation angle in degrees.
        circumference_px: Fitted perimeter in pixels.
        circumference_mm: Physical perimeter in millimeters.
    """
    center_x: float
    center_y: float
    axes_width_px: float   # Full diameter returned by cv2.fitEllipse, not semi-axis radius (a)
    axes_height_px: float  # Full diameter returned by cv2.fitEllipse, not semi-axis radius (b)
    angle_degrees: float
    circumference_px: float
    circumference_mm: float

def fit_fetal_ellipse(contour: np.ndarray, pixel_size_mm: float) -> EllipseParams:
    """
    Fit an ellipse to a contour and compute its circumference in physical millimeters.

    Uses Ramanujan's First Approximation for ellipse perimeter:
    C ≈ pi * [ 3(a + b) - sqrt((3a + b)(a + 3b)) ]

    Args:
        contour: Contour array of shape (N, 1, 2).
        pixel_size_mm: The physical size of one pixel in mm (mm/pixel).

    Returns:
        An EllipseParams dataclass.

    Raises:
        ValueError: If the contour contains fewer than 5 points.
        ValueError: If the pixel size is non-positive.
    """
    if len(contour) < 5:
        raise ValueError(
            f"Ellipse fitting requires a minimum of 5 points. "
            f"Given contour only contains {len(contour)} points."
        )
    if pixel_size_mm <= 0:
        raise ValueError(f"pixel_size_mm must be positive, got {pixel_size_mm}")

    # Fit ellipse: returns ((center_x, center_y), (width, height), angle_degrees)
    (cx, cy), (w, h), angle = cv2.fitEllipse(contour)

    # fitEllipse returns full diameters. Divide by 2 to get semi-axes.
    a = w / 2.0
    b = h / 2.0

    # Ramanujan First Approximation (accurate to < 0.1% for our aspect ratios)
    term1 = 3.0 * (a + b)
    term2 = np.sqrt((3.0 * a + b) * (a + 3.0 * b))
    perimeter_px = np.pi * (term1 - term2)

    # Scale pixel dimensions to millimeters
    perimeter_mm = perimeter_px * pixel_size_mm

    return EllipseParams(
        center_x=cx,
        center_y=cy,
        axes_width_px=w,
        axes_height_px=h,
        angle_degrees=angle,
        circumference_px=perimeter_px,
        circumference_mm=perimeter_mm
    )