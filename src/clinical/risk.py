"""
risk.py
=======
Map a growth percentile to a screening risk band and its semantic presentation.

Bands (per project specification)
---------------------------------
* percentile < 10        -> HIGH    (crimson/rose)  — IUGR screening alert
* 10 <= percentile < 25  -> MEDIUM  (warm amber)    — borderline growth range
* percentile >= 25       -> NORMAL  (vibrant emerald) — expected growth curve

Colors come from ``config.RISK_COLORS`` so the palette lives in one place.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Allow "import config" whether run as a package or from the project root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402


@dataclass(frozen=True)
class RiskAssessment:
    """A classified growth-risk result ready for display."""

    band: str            # "HIGH" | "MEDIUM" | "NORMAL"
    label: str           # short label, e.g. "High Risk"
    headline: str        # one-line clinical framing
    detail: str          # supporting guidance line
    color: str           # primary semantic color (hex)
    accent: str          # secondary semantic color (hex)
    soft: str            # translucent fill for backgrounds
    icon: str            # emoji marker used in badges

    @property
    def streamlit_status(self) -> str:
        """Best-matching st.* alert channel for this band."""
        return {"HIGH": "error", "MEDIUM": "warning", "NORMAL": "success"}[self.band]


def classify(percentile: float) -> RiskAssessment:
    """Classify a growth ``percentile`` (0-100) into a risk assessment."""
    colors = config.RISK_COLORS

    if percentile < config.HIGH_RISK_MAX_PCT:
        c = colors["HIGH"]
        return RiskAssessment(
            band="HIGH",
            label="High Risk",
            headline="Below 10th percentile",
            detail="Screening alert — possible growth restriction (IUGR). Correlate clinically.",
            color=c["solid"], accent=c["accent"], soft=c["soft"], icon="🔴",
        )

    if percentile < config.MEDIUM_RISK_MAX_PCT:
        c = colors["MEDIUM"]
        return RiskAssessment(
            band="MEDIUM",
            label="Medium Risk",
            headline="10th–25th percentile",
            detail="Borderline growth range — monitor closely on serial scans.",
            color=c["solid"], accent=c["accent"], soft=c["soft"], icon="🟡",
        )

    c = colors["NORMAL"]
    return RiskAssessment(
        band="NORMAL",
        label="Normal",
        headline="Above 25th percentile",
        detail="Within the expected growth curve for gestational age.",
        color=c["solid"], accent=c["accent"], soft=c["soft"], icon="🟢",
    )
