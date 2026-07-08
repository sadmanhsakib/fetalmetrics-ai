"""
percentiles.py
==============
Convert a measured head circumference (mm) at a known gestational age into a
z-score and a growth percentile against the Hadlock reference distribution.

The reference distribution is treated as Gaussian at each gestational age,
with the mean and standard deviation supplied by ``reference_hadlock``.  The
normal CDF is evaluated using ``math.erf`` so there is no SciPy dependency;
the result is numerically identical to ``scipy.stats.norm.cdf`` for all
practically relevant z-score values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import reference_hadlock as ref


def normal_cdf(z: float) -> float:
    """Evaluate the standard-normal cumulative distribution function Φ(z).

    Implemented in terms of the error function from the standard library to
    avoid any external dependencies.  Numerically equivalent to
    ``0.5 * erfc(-z / sqrt(2))``.

    Parameters
    ----------
    z:
        Standard-normal variate.

    Returns
    -------
    float
        Cumulative probability P(Z ≤ z) in the range [0, 1].
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class PercentileResult:
    """Outcome of placing one HC measurement on the Hadlock growth curve.

    Attributes
    ----------
    hc_mm:
        Measured head circumference in millimetres.
    ga_weeks:
        Gestational age in weeks at which the measurement was taken.
    mean_mm:
        Reference population mean HC at this gestational age.
    sd_mm:
        Reference population standard deviation of HC at this gestational age.
    z_score:
        Signed standard-score of the measurement relative to the reference
        distribution.  Negative values indicate below-average HC.
    percentile:
        Growth percentile clamped to [0.1, 99.9] for display stability.
    percentile_raw:
        Unclamped growth percentile, retained for downstream calculations
        that require the true tail value.
    """

    hc_mm: float
    ga_weeks: float
    mean_mm: float
    sd_mm: float
    z_score: float
    percentile: float        # 0–100, clamped to (0.1, 99.9) for display
    percentile_raw: float    # Unclamped 0–100

    @property
    def percentile_label(self) -> str:
        """Format the percentile as a human-readable ordinal string.

        Examples: ``"18.4th"``, ``"21st"``, ``"2nd"``, ``"3rd"``.

        Non-integer percentiles always use the ``"th"`` suffix.  Integer
        values use the correct English ordinal suffix, with special handling
        for the 11th–13th range which takes ``"th"`` regardless of last digit.
        """
        p = self.percentile

        # If the value is very close to an integer, format it as an ordinal.
        if abs(p - round(p)) < 0.05:
            val_int = int(round(p))
            # The 11th–13th are irregular exceptions to the standard suffix rules.
            if 10 <= val_int % 100 <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(val_int % 10, "th")
            return f"{val_int}{suffix}"

        # Decimal percentiles always use "th".
        return f"{p:.1f}th"


def evaluate(hc_mm: float, ga_weeks: float) -> PercentileResult:
    """Compute the z-score and growth percentile for an HC measurement.

    Parameters
    ----------
    hc_mm:
        Measured head circumference in millimetres.
    ga_weeks:
        Gestational age in weeks (14–40).

    Returns
    -------
    PercentileResult
        Complete percentile evaluation, including the reference distribution
        parameters used in the computation.
    """
    mean = ref.mean_hc_mm(ga_weeks)
    sd = ref.sd_hc_mm(ga_weeks)

    # Guard against a zero SD (should not occur given the SD floor, but
    # handled defensively to prevent a ZeroDivisionError).
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
