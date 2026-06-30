"""Module for fetal head contour extraction."""
import cv2
import numpy as np

def extract_largest_contour(mask: np.ndarray, min_area_px: float = 1000.0) -> np.ndarray:
    """
    Extract the largest external contour from a binary mask.

    Args:
        mask: Grayscale binary mask array, shape (H, W). Value range [0, 255] or [0, 1].
        min_area_px: Minimum contour area in pixels to reject noise.

    Returns:
        The largest contour as a numpy array of shape (N, 1, 2).

    Raises:
        ValueError: If no contours are found or the largest is smaller than min_area_px.
    """
    # Ensure mask is uint8 in 0-255 scale
    if mask.max() == 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours detected in the binary mask.")

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < min_area_px:
        raise ValueError(
            f"Largest detected region is too small ({area:.1f} px). "
            f"Expected at least {min_area_px} px."
        )

    return largest_contour