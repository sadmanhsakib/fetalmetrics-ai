"""
pixel_size.py
=============
Resolve the physical scale of an ultrasound image: millimetres per pixel.

Two sources, in priority order:
1. **HC18 metadata** — the Grand-Challenge CSVs map each image filename to its
   ``pixel size(mm)``. If the uploaded file's name matches, we use that.
2. **Manual override** — a value entered in the UI, used for arbitrary images
   that are not part of the HC18 distribution.

A sensible default is offered only as a last resort and is clearly flagged in
the returned source so the UI can warn the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import config


@dataclass(frozen=True)
class Calibration:
    """The scale used to convert pixel measurements to millimetres."""

    mm_per_px: float
    source: str          # "hc18" | "manual" | "default"
    matched_name: str | None = None

    @property
    def source_label(self) -> str:
        return {
            "hc18": "HC18 metadata",
            "manual": "Manual entry",
            "default": "Fallback default",
        }.get(self.source, self.source)

    @property
    def is_trustworthy(self) -> bool:
        return self.source in ("hc18", "manual")


def _resolve_collision(filename: str, image_rgb: np.ndarray) -> str | None:
    """Compare image_rgb with local raw images to identify training vs test set."""
    from PIL import Image

    # Paths to local files
    train_path = config.DATA_DIR / "training_set" / filename
    test_path = config.DATA_DIR / "test_set" / filename

    best_dataset = None
    min_diff = float("inf")

    for dataset_name, path in [("training", train_path), ("test", test_path)]:
        if path.exists():
            try:
                # Load local raw image
                local_img = np.array(Image.open(path).convert("RGB"))
                if local_img.shape == image_rgb.shape:
                    diff = np.mean(np.abs(image_rgb.astype(float) - local_img.astype(float)))
                    if diff < min_diff:
                        min_diff = diff
                        best_dataset = dataset_name
            except Exception:
                continue

    # If the match is very close (e.g. mean pixel diff < 5.0), return the matching dataset
    if min_diff < 5.0:
        return best_dataset
    return None


def _normalize(name: str) -> str:
    """Filename key used for matching (basename, lower-case)."""
    return Path(str(name)).name.strip().lower()


@lru_cache(maxsize=1)
def _load_hc18_table() -> dict[str, list[tuple[str, float]]]:
    """Build {filename -> [(dataset_name, mm_per_px)]} from any available HC18 CSVs.

    Pandas is imported lazily so importing this module never forces the
    dependency (keeps unit tests light).
    """
    table: dict[str, list[tuple[str, float]]] = {}
    try:
        import pandas as pd
    except Exception:
        return table

    # Match each CSV file to a dataset identifier ("training", "test")
    csv_mapping = {}
    for csv_path in config.HC18_PIXEL_SIZE_CSVS:
        name_lower = csv_path.name.lower()
        if "training" in name_lower:
            csv_mapping["training"] = csv_path
        elif "test" in name_lower:
            csv_mapping["test"] = csv_path

    for dataset_name, csv_path in csv_mapping.items():
        if not Path(csv_path).exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        # Locate the filename and pixel-size columns tolerantly.
        cols = {c.lower().strip(): c for c in df.columns}
        fname_col = next((cols[k] for k in cols if "filename" in k or k == "file"), None)
        px_col = next((cols[k] for k in cols if "pixel" in k and "size" in k), None)
        if fname_col is None or px_col is None:
            continue

        for _, row in df[[fname_col, px_col]].dropna().iterrows():
            try:
                name_key = _normalize(row[fname_col])
                val = float(row[px_col])
                if name_key not in table:
                    table[name_key] = []
                table[name_key].append((dataset_name, val))
            except (TypeError, ValueError):
                continue

    return table


def lookup_hc18(filename: str, image_rgb: np.ndarray | None = None) -> float | None:
    """Return the HC18 mm/pixel for ``filename`` if present, else ``None``.

    Uses image_rgb content matching to resolve conflicts if filename exists in both splits.
    """
    candidates = _load_hc18_table().get(_normalize(filename))
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][1]

    # Collision! Resolve using image content if available
    if image_rgb is not None:
        dataset = _resolve_collision(filename, image_rgb)
        if dataset:
            for d, px_size in candidates:
                if d == dataset:
                    return px_size

    # Fallback to the first candidate (training set)
    return candidates[0][1]


def resolve(
    filename: str | None,
    image_rgb: np.ndarray | None = None,
    manual_mm_per_px: float | None = None,
    prefer_manual: bool = False,
) -> Calibration:
    """Resolve the calibration to use for an image.

    Parameters
    ----------
    filename:
        Uploaded file name, used to query the HC18 table.
    image_rgb:
        Optional uploaded image contents to resolve set collisions.
    manual_mm_per_px:
        Optional user-entered scale.
    prefer_manual:
        If True, a valid manual value wins even when HC18 metadata exists
        (lets the user deliberately override auto-calibration).
    """
    manual_ok = manual_mm_per_px is not None and manual_mm_per_px > 0

    if prefer_manual and manual_ok:
        return Calibration(mm_per_px=float(manual_mm_per_px), source="manual")

    if filename:
        hc18 = lookup_hc18(filename, image_rgb)
        if hc18 and hc18 > 0:
            return Calibration(mm_per_px=hc18, source="hc18", matched_name=_normalize(filename))

    if manual_ok:
        return Calibration(mm_per_px=float(manual_mm_per_px), source="manual")

    return Calibration(mm_per_px=float(config.DEFAULT_PIXEL_SIZE_MM), source="default")


def hc18_available() -> bool:
    """True when at least one HC18 pixel-size row was loaded."""
    return len(_load_hc18_table()) > 0

