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

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402


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


def _normalize(name: str) -> str:
    """Filename key used for matching (basename, lower-case)."""
    return Path(str(name)).name.strip().lower()


@lru_cache(maxsize=1)
def _load_hc18_table() -> dict[str, float]:
    """Build {filename -> mm_per_px} from any available HC18 CSVs.

    Pandas is imported lazily so importing this module never forces the
    dependency (keeps unit tests light).
    """
    table: dict[str, float] = {}
    try:
        import pandas as pd
    except Exception:
        return table

    for csv_path in config.HC18_PIXEL_SIZE_CSVS:
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
                table[_normalize(row[fname_col])] = float(row[px_col])
            except (TypeError, ValueError):
                continue

    return table


def lookup_hc18(filename: str) -> float | None:
    """Return the HC18 mm/pixel for ``filename`` if present, else ``None``."""
    return _load_hc18_table().get(_normalize(filename))


def resolve(
    filename: str | None,
    manual_mm_per_px: float | None = None,
    prefer_manual: bool = False,
) -> Calibration:
    """Resolve the calibration to use for an image.

    Parameters
    ----------
    filename:
        Uploaded file name, used to query the HC18 table.
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
        hc18 = lookup_hc18(filename)
        if hc18 and hc18 > 0:
            return Calibration(mm_per_px=hc18, source="hc18", matched_name=_normalize(filename))

    if manual_ok:
        return Calibration(mm_per_px=float(manual_mm_per_px), source="manual")

    return Calibration(mm_per_px=float(config.DEFAULT_PIXEL_SIZE_MM), source="default")


def hc18_available() -> bool:
    """True when at least one HC18 pixel-size row was loaded."""
    return len(_load_hc18_table()) > 0
