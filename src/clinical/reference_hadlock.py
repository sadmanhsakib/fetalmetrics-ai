"""
reference_hadlock.py
====================
Head-circumference-for-gestational-age reference used to place a measured HC on
the fetal growth distribution.

Design intent
-------------
The percentile engine only needs two functions of gestational age (GA): the
population **mean HC** and its **standard deviation**. Both are exposed here as
small, transparent, *swappable* objects so the clinical logic is fully
auditable and easy to replace with your own calibrated Hadlock coefficients or
an INTERGROWTH-21st table.

* ``_MEAN_HC_MM`` — mean HC (mm) per completed gestational week, consistent with
  the Hadlock composite fetal-biometry charts in the 14-40 week range.
* ``sd_hc_mm``    — a documented linear SD model anchored to published HC
  variability (≈5 mm at 14 w rising to ≈15 mm at 40 w).

Intermediate (half-week) values are obtained by linear interpolation, giving a
smooth, monotonic reference curve.

References
----------
Hadlock FP, Deter RL, Harrist RB, Park SK. "Estimating fetal age: computer-
assisted analysis of multiple fetal growth parameters." Radiology 1984;
152:497-501.
Chitty LS et al. "Charts of fetal size: 2. Head measurements." BJOG 1994;
101:35-43. (Comparable in the normal range; useful cross-check.)

NOTE
----
The coefficients below are a faithful, literature-consistent reference intended
for demonstration. For a submission or deployment, verify them against your own
Hadlock fit and cite the exact source; swapping in new numbers here changes the
whole app's percentile behaviour with no other code changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Mean HC (mm) by completed gestational week — Hadlock composite reference.
_MEAN_HC_MM: dict[int, float] = {
    14: 103.0, 15: 115.0, 16: 127.0, 17: 138.0, 18: 150.0, 19: 162.0,
    20: 174.0, 21: 185.0, 22: 197.0, 23: 208.0, 24: 219.0, 25: 230.0,
    26: 241.0, 27: 251.0, 28: 262.0, 29: 271.0, 30: 281.0, 31: 290.0,
    32: 299.0, 33: 307.0, 34: 315.0, 35: 322.0, 36: 329.0, 37: 335.0,
    38: 340.0, 39: 345.0, 40: 349.0,
}

# Standard-deviation model: SD(GA) = _SD_INTERCEPT + _SD_SLOPE * GA  (mm).
# Anchored to ~5 mm at 14 w and ~15 mm at 40 w.
_SD_SLOPE: float = 10.0 / 26.0          # ≈ 0.3846 mm / week
_SD_INTERCEPT: float = 5.0 - _SD_SLOPE * 14.0
_SD_FLOOR_MM: float = 3.0

# z-scores for common reference percentiles (standard normal).
Z_P10: float = -1.2815515594
Z_P50: float = 0.0
Z_P90: float = 1.2815515594

GA_MIN_WEEKS: float = float(min(_MEAN_HC_MM))
GA_MAX_WEEKS: float = float(max(_MEAN_HC_MM))

CITATION: str = (
    "Hadlock FP et al., Radiology 1984;152:497-501 (HC-for-GA composite). "
    "SD modelled linearly from published HC variability."
)


def _clamp_ga(ga_weeks: float) -> float:
    """Keep GA inside the supported reference range."""
    return max(GA_MIN_WEEKS, min(GA_MAX_WEEKS, float(ga_weeks)))


def mean_hc_mm(ga_weeks: float) -> float:
    """Population mean HC (mm) at the given gestational age (weeks).

    Linear interpolation between the tabulated integer-week means.
    """
    ga = _clamp_ga(ga_weeks)
    lo = int(math.floor(ga))
    hi = int(math.ceil(ga))
    if lo == hi:
        return _MEAN_HC_MM[lo]
    frac = ga - lo
    return _MEAN_HC_MM[lo] * (1.0 - frac) + _MEAN_HC_MM[hi] * frac


def sd_hc_mm(ga_weeks: float) -> float:
    """Standard deviation of HC (mm) at the given gestational age (weeks)."""
    ga = _clamp_ga(ga_weeks)
    return max(_SD_FLOOR_MM, _SD_INTERCEPT + _SD_SLOPE * ga)


@dataclass(frozen=True)
class ReferenceSummary:
    """Snapshot of the reference distribution at one gestational age."""

    ga_weeks: float
    mean_mm: float
    sd_mm: float
    p10_mm: float
    p50_mm: float
    p90_mm: float


def reference_summary(ga_weeks: float) -> ReferenceSummary:
    """Mean, SD and the 10th/50th/90th reference HC values (mm) at a given GA."""
    mean = mean_hc_mm(ga_weeks)
    sd = sd_hc_mm(ga_weeks)
    return ReferenceSummary(
        ga_weeks=float(ga_weeks),
        mean_mm=mean,
        sd_mm=sd,
        p10_mm=mean + Z_P10 * sd,
        p50_mm=mean + Z_P50 * sd,
        p90_mm=mean + Z_P90 * sd,
    )
