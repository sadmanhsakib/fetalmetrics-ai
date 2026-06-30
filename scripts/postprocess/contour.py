"""Module for fetal head contour extraction."""
import os
import random

import cv2
import numpy as np
from pyprojroot import here

IMG_DIR = here("data/preprocessed/fastai/images/train/")


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


def main():
    sample_img = os.path.join(IMG_DIR, random.choice(os.listdir(IMG_DIR)))
    mask_path = str(sample_img).replace("images", "masks")

    if not os.path.exists(mask_path):
        print(f"Corresponding mask not found at: {mask_path}")
        return

    print(f"Loading actual mask for: {os.path.basename(sample_img)}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to read mask from: {mask_path}")
        return

    try:
        largest_contour = extract_largest_contour(mask)
        print(f"Successfully extracted largest contour.")
        print(f"Contour shape: {largest_contour.shape}")
        print(f"Contour area: {cv2.contourArea(largest_contour):.1f} px")

        img = cv2.imread(sample_img)
        cv2.drawContours(img, largest_contour, -1, (0, 255, 0), 2)

        cv2.imshow("Contours", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
