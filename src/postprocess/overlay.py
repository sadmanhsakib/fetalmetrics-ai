"""
overlay.py
==========
Render the segmentation result on top of the ultrasound image:

* a translucent mask fill (optional, subtle),
* the fitted ellipse contour in electric cyan,
* a crosshair along the ellipse's major/minor axes to confirm placement,
* a small centroid marker.

Colors come from ``config.OVERLAY`` (RGB). OpenCV works in BGR, so we convert
at the boundary and return a clean RGB uint8 image ready for ``st.image``.
"""

from __future__ import annotations

import numpy as np

import config
from postprocess.ellipse import HCResult


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce any grayscale, RGB, or RGBA array to a contiguous RGB uint8 image.
    
    Parameters
    ----------
    image:
        Input image array to be coerced.
        
    Returns
    -------
    numpy.ndarray
        RGB uint8 contiguous image array.
    """
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        if img.size > 0 and img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = np.squeeze(img, axis=2)
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
    return np.ascontiguousarray(img)


def render(
    image: np.ndarray,
    result: HCResult,
    mask: np.ndarray | None = None,
    draw_mask_fill: bool = True,
    draw_crosshair: bool = True,
) -> np.ndarray:
    """Return an RGB image superimposed with the ellipse, crosshair and optional mask fill.

    Parameters
    ----------
    image:
        Original ultrasound image (grayscale or RGB).
    result:
        Output of ``ellipse.measure_hc``, which carries the exact fitted ellipse.
    mask:
        Optional binary skull mask, used only for the translucent fill.
    draw_mask_fill:
        Whether to draw a translucent overlay inside the segmentation mask.
    draw_crosshair:
        Whether to draw crosshair lines indicating the major and minor axes.

    Returns
    -------
    numpy.ndarray
        Composite RGB image array ready for display.
    """
    import cv2

    rgb = _to_rgb_uint8(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    ell_bgr = tuple(reversed(config.OVERLAY["ellipse_rgb"]))
    cross_bgr = tuple(reversed(config.OVERLAY["crosshair_rgb"]))
    fill_bgr = tuple(reversed(config.OVERLAY["mask_fill_rgb"]))

    (cx, cy), (axis_a, axis_b), angle = result.ellipse_cv
    center = (int(round(cx)), int(round(cy)))
    axes_half = (int(round(axis_a / 2.0)), int(round(axis_b / 2.0)))
    angle_i = int(round(angle))

    # 1) translucent mask fill --------------------------------------------- #
    if draw_mask_fill and mask is not None:
        overlay = bgr.copy()
        m = (np.asarray(mask) > 0)
        overlay[m] = fill_bgr
        alpha = float(config.OVERLAY["mask_fill_alpha"])
        bgr = cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0.0)

    # 2) crosshair along the ellipse axes ---------------------------------- #
    if axes_half[0] > 0 and axes_half[1] > 0:
        if draw_crosshair:
            # Four axis vertices via the ellipse's own parametric frame.
            pts = cv2.ellipse2Poly(center, axes_half, angle_i, 0, 360, 90)
            if len(pts) >= 4:
                t = int(config.OVERLAY["crosshair_thickness"])
                cv2.line(bgr, tuple(pts[0]), tuple(pts[2]), cross_bgr, t, cv2.LINE_AA)
                cv2.line(bgr, tuple(pts[1]), tuple(pts[3]), cross_bgr, t, cv2.LINE_AA)

        # 3) ellipse contour --------------------------------------------------- #
        cv2.ellipse(
            bgr, center, axes_half, angle_i, 0, 360,
            ell_bgr, int(config.OVERLAY["ellipse_thickness"]), cv2.LINE_AA,
        )

        # 4) centroid marker --------------------------------------------------- #
        cv2.circle(bgr, center, 3, ell_bgr, -1, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
