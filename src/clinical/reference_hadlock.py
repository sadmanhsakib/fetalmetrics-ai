"""
reference_hadlock.py
====================
Head-circumference-for-gestational-age reference values used to place a
measured HC on the population growth distribution.

Design
------
The percentile engine requires two functions of gestational age (GA): the
population mean HC and its standard deviation.  Both are exposed here as
compact, transparent, swappable objects so the clinical logic is fully
auditable and straightforward to replace with calibrated Hadlock coefficients
from your own demographic or with an INTERGROWTH-21st table.

Mean HC (``_MEAN_HC_MM``)
    Mean HC in mm at each completed gestational week, consistent with the
    Hadlock composite fetal-biometry charts across the 14–40 week range.
    Intermediate (half-week) values are obtained by linear interpolation,
    giving a smooth, monotonic reference curve.

Standard deviation (``sd_hc_mm``)
    A linear SD model anchored to published HC variability:
    approximately 5 mm at 14 weeks, rising to approximately 15 mm at 40 weeks.
    A floor of ``_SD_FLOOR_MM`` prevents the model from producing unrealistically
    small SD values near the lower gestational-age boundary.

References
----------
Hadlock FP, Deter RL, Harrist RB, Park SK. "Estimating fetal age:
computer-assisted analysis of multiple fetal growth parameters."
Radiology 1984;152:497–501.

Chitty LS et al. "Charts of fetal size: 2. Head measurements."
BJOG 1994;101:35–43. (Comparable in the normal range; used as a
cross-reference for the tabulated means.)

.. note::
   The coefficients below are a literature-consistent reference intended
   for demonstration.  For production use or clinical submission, verify
   them against your own Hadlock fit and cite the exact source.  Swapping
   in new numbers here changes the entire application's percentile behaviour
   with no other code changes required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

# Mean HC (mm) at each completed gestational week — Hadlock composite reference.
_MEAN_HC_MM: dict[int, float] = {
    14: 103.0, 15: 115.0, 16: 127.0, 17: 138.0, 18: 150.0, 19: 162.0,
    20: 174.0, 21: 185.0, 22: 197.0, 23: 208.0, 24: 219.0, 25: 230.0,
    26: 241.0, 27: 251.0, 28: 262.0, 29: 271.0, 30: 281.0, 31: 290.0,
    32: 299.0, 33: 307.0, 34: 315.0, 35: 322.0, 36: 329.0, 37: 335.0,
    38: 340.0, 39: 345.0, 40: 349.0,
}

# ---------------------------------------------------------------------------
# Standard-deviation model: SD(GA) = _SD_INTERCEPT + _SD_SLOPE × GA  (mm)
# Anchored to ≈5 mm at 14 w and ≈15 mm at 40 w, with a hard floor to avoid
# implausibly small values at the lower gestational-age boundary.
# ---------------------------------------------------------------------------
_SD_SLOPE: float = 10.0 / 26.0              # ≈ 0.3846 mm/week
_SD_INTERCEPT: float = 5.0 - _SD_SLOPE * 14.0
_SD_FLOOR_MM: float = 3.0

# ---------------------------------------------------------------------------
# Standard-normal z-scores for reference percentiles.
# ---------------------------------------------------------------------------
Z_P10: float = -1.2815515594
Z_P50: float = 0.0
Z_P90: float = 1.2815515594

# ---------------------------------------------------------------------------
# GA range derived from the look-up table — prevents callers from needing
# to hard-code the same constants.
# ---------------------------------------------------------------------------
GA_MIN_WEEKS: float = float(min(_MEAN_HC_MM))
GA_MAX_WEEKS: float = float(max(_MEAN_HC_MM))

CITATION: str = (
    "Hadlock FP et al., Radiology 1984;152:497–501 (HC-for-GA composite). "
    "SD modelled linearly from published HC variability."
)


# ---------------------------------------------------------------------------
# Reference functions
# ---------------------------------------------------------------------------

def _clamp_ga(ga_weeks: float) -> float:
    """Clamp ``ga_weeks`` to the supported reference range [GA_MIN, GA_MAX]."""
    return max(GA_MIN_WEEKS, min(GA_MAX_WEEKS, float(ga_weeks)))


def mean_hc_mm(ga_weeks: float) -> float:
    """Return the population mean HC in mm at the given gestational age.

    Values are linearly interpolated between adjacent integer-week entries in
    the Hadlock table, giving a smooth, monotonic curve.  Gestational ages
    outside the supported range are clamped to the nearest boundary.

    Parameters
    ----------
    ga_weeks:
        Gestational age in weeks (will be clamped to [14, 40]).

    Returns
    -------
    float
        Population mean head circumference in millimetres.
    """
    ga = _clamp_ga(ga_weeks)
    lo = int(math.floor(ga))
    hi = int(math.ceil(ga))
    if lo == hi:
        return _MEAN_HC_MM[lo]
    frac = ga - lo
    return _MEAN_HC_MM[lo] * (1.0 - frac) + _MEAN_HC_MM[hi] * frac


def sd_hc_mm(ga_weeks: float) -> float:
    """Return the standard deviation of HC in mm at the given gestational age.

    The SD is modelled as a linear function of GA, anchored to published
    variability data and floored at ``_SD_FLOOR_MM`` to remain physically
    meaningful across the full supported range.

    Parameters
    ----------
    ga_weeks:
        Gestational age in weeks (will be clamped to [14, 40]).

    Returns
    -------
    float
        Population standard deviation of head circumference in millimetres.
    """
    ga = _clamp_ga(ga_weeks)
    return max(_SD_FLOOR_MM, _SD_INTERCEPT + _SD_SLOPE * ga)


@dataclass(frozen=True)
class ReferenceSummary:
    """Snapshot of the HC reference distribution at one gestational age.

    Attributes
    ----------
    ga_weeks:
        Gestational age for which this summary was computed.
    mean_mm:
        Population mean HC in millimetres.
    sd_mm:
        Population standard deviation of HC in millimetres.
    p10_mm, p50_mm, p90_mm:
        10th, 50th, and 90th reference percentile HC values in millimetres,
        derived from the Gaussian model.
    """

    ga_weeks: float
    mean_mm: float
    sd_mm: float
    p10_mm: float
    p50_mm: float
    p90_mm: float


def reference_summary(ga_weeks: float) -> ReferenceSummary:
    """Return the mean, SD, and 10th/50th/90th percentile HC values at a given GA.

    Parameters
    ----------
    ga_weeks:
        Gestational age in weeks.

    Returns
    -------
    ReferenceSummary
        Complete distributional snapshot at the specified gestational age.
    """
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
