"""
pixel_size.py
=============
Resolve the physical scale of an ultrasound image: millimetres per pixel.

The calibration resolver applies the following priority order:

1. **HC18 metadata** — the Grand Challenge CSVs map each image filename to its
   ``pixel size(mm)`` value.  When the uploaded filename matches a record, the
   dataset-derived scale is used.
2. **Manual override** — a value entered in the UI, used for images that are
   not part of the HC18 distribution or where the operator wishes to correct
   the automatic lookup.
3. **Fallback default** — ``config.DEFAULT_PIXEL_SIZE_MM`` is used as a last
   resort and is explicitly flagged in the returned ``Calibration`` object so
   the UI can display a prominent warning.

When the same filename appears in both the training and test splits (a
collision), ``_resolve_collision`` compares pixel intensity with local copies
of the raw images to select the correct record.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import config


@dataclass(frozen=True)
class Calibration:
    """The resolved physical scale for a single image.

    Attributes
    ----------
    mm_per_px:
        Physical size of one pixel in millimetres.
    source:
        Resolution path.  One of ``"hc18"`` (dataset metadata), ``"manual"``
        (operator entry), or ``"default"`` (fallback constant).
    matched_name:
        Normalised filename that produced the HC18 match, or ``None`` when
        the source is not ``"hc18"``.
    """

    mm_per_px: float
    source: str                  # "hc18" | "manual" | "default"
    matched_name: str | None = None

    @property
    def source_label(self) -> str:
        """Human-readable description of the calibration source."""
        return {
            "hc18": "HC18 metadata",
            "manual": "Manual entry",
            "default": "Fallback default",
        }.get(self.source, self.source)

    @property
    def is_trustworthy(self) -> bool:
        """``True`` when the scale comes from a reliable source.

        A ``False`` value indicates that the fallback default is in use and
        that the HC measurement does not carry physical units.
        """
        return self.source in ("hc18", "manual")


def _resolve_collision(filename: str, image_rgb: np.ndarray) -> str | None:
    """Disambiguate a filename that appears in both HC18 training and test CSVs.

    Loads the corresponding local raw images from both dataset directories and
    computes the mean absolute pixel difference against the uploaded image.
    The dataset whose local file is closest in pixel content is selected.

    Parameters
    ----------
    filename:
        Original uploaded filename (basename only).
    image_rgb:
        RGB uint8 pixel array of the uploaded image.

    Returns
    -------
    str | None
        ``"training"`` or ``"test"`` if a confident match is found (mean
        absolute difference < 5.0), otherwise ``None``.
    """
    from PIL import Image

    train_path = config.DATA_DIR / "training_set" / filename
    test_path = config.DATA_DIR / "test_set" / filename

    best_dataset: str | None = None
    min_diff = float("inf")

    for dataset_name, path in [("training", train_path), ("test", test_path)]:
        if not path.exists():
            continue
        try:
            local_img = np.array(Image.open(path).convert("RGB"))
            if local_img.shape == image_rgb.shape:
                diff = np.mean(
                    np.abs(image_rgb.astype(float) - local_img.astype(float))
                )
                if diff < min_diff:
                    min_diff = diff
                    best_dataset = dataset_name
        except Exception:
            continue

    # Accept the nearest match only when the pixel difference is
    # unambiguously small; a threshold of 5.0 tolerates minor JPEG
    # artefacts while rejecting genuinely different images.
    return best_dataset if min_diff < 5.0 else None


def _normalize(name: str) -> str:
    """Return the canonical lookup key for a filename (basename, lower-case)."""
    return Path(str(name)).name.strip().lower()


@lru_cache(maxsize=1)
def _load_hc18_table() -> dict[str, list[tuple[str, float]]]:
    """Parse all available HC18 CSVs into a filename-keyed look-up table.

    The returned structure maps each normalised filename to a list of
    ``(dataset_name, mm_per_px)`` tuples — one per CSV that contains a
    matching entry.  A list is used (rather than a single value) so that
    multi-dataset collisions can be resolved downstream.

    Pandas is imported lazily to keep this module importable in lightweight
    environments where only the pure-Python calibration logic is exercised.

    Returns
    -------
    dict[str, list[tuple[str, float]]]
        Mapping: normalised filename → [(dataset_name, mm_per_px), …]
    """
    table: dict[str, list[tuple[str, float]]] = {}
    try:
        import pandas as pd
    except Exception:
        return table

    # Associate each CSV path with a dataset identifier derived from its name.
    csv_mapping: dict[str, Path] = {}
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

        # Column detection is intentionally tolerant of minor naming variations
        # (e.g. extra whitespace, different capitalisation).
        cols = {c.lower().strip(): c for c in df.columns}
        fname_col = next(
            (cols[k] for k in cols if "filename" in k or k == "file"), None
        )
        px_col = next(
            (cols[k] for k in cols if "pixel" in k and "size" in k), None
        )
        if fname_col is None or px_col is None:
            continue

        for _, row in df[[fname_col, px_col]].dropna().iterrows():
            try:
                name_key = _normalize(row[fname_col])
                val = float(row[px_col])
                table.setdefault(name_key, []).append((dataset_name, val))
            except (TypeError, ValueError):
                continue

    return table


def lookup_hc18(filename: str, image_rgb: np.ndarray | None = None) -> float | None:
    """Return the HC18 mm/pixel scale for ``filename``, or ``None`` if absent.

    When ``filename`` appears in both the training and test CSVs, the correct
    record is selected by comparing pixel content with local raw images via
    ``_resolve_collision``.  If no local images are available for comparison,
    the training-set record is returned as a conservative fallback.

    Parameters
    ----------
    filename:
        Uploaded image filename (basename only).
    image_rgb:
        Optional RGB uint8 array of the uploaded image, used only to resolve
        filename collisions between dataset splits.

    Returns
    -------
    float | None
        Pixel scale in mm/pixel, or ``None`` if the filename is not found in
        any loaded HC18 CSV.
    """
    candidates = _load_hc18_table().get(_normalize(filename))
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][1]

    # Filename appears in both splits — attempt image-content disambiguation.
    if image_rgb is not None:
        dataset = _resolve_collision(filename, image_rgb)
        if dataset:
            for d, px_size in candidates:
                if d == dataset:
                    return px_size

    # Fall back to the training-set record when disambiguation is not possible.
    return candidates[0][1]


def resolve(
    filename: str | None,
    image_rgb: np.ndarray | None = None,
    manual_mm_per_px: float | None = None,
    prefer_manual: bool = False,
) -> Calibration:
    """Resolve the calibration scale for a given image, applying the priority chain.

    Parameters
    ----------
    filename:
        Uploaded filename, used to query the HC18 look-up table.  Pass
        ``None`` to skip the HC18 lookup entirely.
    image_rgb:
        Optional pixel array of the uploaded image, used to resolve filename
        collisions between training and test splits.
    manual_mm_per_px:
        Pixel scale entered by the operator in the UI.  Values of ``None`` or
        ≤ 0 are treated as absent.
    prefer_manual:
        When ``True``, a valid manual value takes precedence over any HC18
        metadata match.  This lets the operator explicitly override automatic
        calibration.

    Returns
    -------
    Calibration
        The resolved calibration, with ``source`` set to indicate which
        resolution path was taken.
    """
    manual_ok = manual_mm_per_px is not None and manual_mm_per_px > 0

    if prefer_manual and manual_ok:
        return Calibration(mm_per_px=float(manual_mm_per_px), source="manual")

    if filename:
        hc18 = lookup_hc18(filename, image_rgb)
        if hc18 and hc18 > 0:
            return Calibration(
                mm_per_px=hc18,
                source="hc18",
                matched_name=_normalize(filename),
            )

    if manual_ok:
        return Calibration(mm_per_px=float(manual_mm_per_px), source="manual")

    return Calibration(mm_per_px=float(config.DEFAULT_PIXEL_SIZE_MM), source="default")


def hc18_available() -> bool:
    """Return ``True`` when at least one HC18 pixel-size record has been loaded."""
    return len(_load_hc18_table()) > 0
