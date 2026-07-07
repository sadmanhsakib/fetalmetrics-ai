"""
percentiles.py
==============
Convert a measured head circumference (mm) at a known gestational age into a
z-score and a growth percentile against the reference distribution.

The reference distribution is treated as Gaussian at each gestational age
(mean and SD supplied by ``reference_hadlock``). The normal CDF is evaluated
with ``math.erf`` so there is **no SciPy dependency**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import reference_hadlock as ref


def normal_cdf(z: float) -> float:
    """Standard-normal cumulative distribution function Φ(z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class PercentileResult:
    """Outcome of placing one HC measurement on the growth curve."""

    hc_mm: float
    ga_weeks: float
    mean_mm: float
    sd_mm: float
    z_score: float
    percentile: float          # 0-100, clamped to (0.1, 99.9) for display
    percentile_raw: float      # unclamped 0-100

    @property
    def percentile_label(self) -> str:
        """Human-friendly percentile, e.g. '18.4th' or '21st'."""
        p = self.percentile
        # If it's very close to an integer, format as integer and use correct suffix
        if abs(p - round(p)) < 0.05:
            val_int = int(round(p))
            if 10 <= val_int % 100 <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(val_int % 10, "th")
            return f"{val_int}{suffix}"
        else:
            # Decimals always take "th"
            return f"{p:.1f}th"


def evaluate(hc_mm: float, ga_weeks: float) -> PercentileResult:
    """Compute z-score and percentile for ``hc_mm`` at ``ga_weeks``.

    Parameters
    ----------
    hc_mm:
        Measured head circumference in millimetres.
    ga_weeks:
        Gestational age in weeks (14-40).
    """
    mean = ref.mean_hc_mm(ga_weeks)
    sd = ref.sd_hc_mm(ga_weeks)
    z = (hc_mm - mean) / sd if sd > 0 else 0.0
    pct_raw = normal_cdf(z) * 100.0
    pct = min(99.9, max(0.1, pct_raw))
    return PercentileResult(
        hc_mm=float(hc_mm),
        ga_weeks=float(ga_weeks),
        mean_mm=mean,
        sd_mm=sd,
        z_score=z,
        percentile=pct,
        percentile_raw=pct_raw,
    )
