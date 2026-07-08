"""
risk.py
=======
Map a growth percentile to a clinical screening risk band with semantic
presentation tokens.

Risk bands (per project specification)
---------------------------------------
* percentile < 10        → HIGH    (crimson)   — IUGR screening alert
* 10 ≤ percentile < 25  → MEDIUM  (amber)     — borderline growth range
* percentile ≥ 25       → NORMAL  (emerald)   — within expected growth curve

All color tokens are sourced from ``config.RISK_COLORS`` so the palette is
defined in a single location and shared between the risk logic, the UI
components, and the methodology documentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class RiskAssessment:
    """A classified growth-risk result with all tokens needed for display.

    Attributes
    ----------
    band:
        Canonical risk tier: ``"HIGH"``, ``"MEDIUM"``, or ``"NORMAL"``.
    label:
        Short display label, e.g. ``"High Risk"``.
    headline:
        One-line clinical framing shown as the card title.
    detail:
        Supporting guidance text shown below the headline.
    color:
        Primary semantic hex color for the risk tier (solid variant).
    accent:
        Secondary semantic hex color (darker variant, used for borders and
        emphasis).
    soft:
        Translucent fill color (``rgba(…)`` string) for card backgrounds.
    icon:
        Emoji status marker used in compact badges and notifications.
    """

    band: str       # "HIGH" | "MEDIUM" | "NORMAL"
    label: str
    headline: str
    detail: str
    color: str      # Primary solid hex color
    accent: str     # Darker accent hex color
    soft: str       # Translucent rgba fill
    icon: str       # Emoji status marker

    @property
    def streamlit_status(self) -> str:
        """Map the risk band to the corresponding Streamlit alert channel.

        Returns
        -------
        str
            One of ``"error"``, ``"warning"``, or ``"success"``, suitable for
            use as the ``type`` argument of ``st.status`` or similar widgets.
        """
        return {"HIGH": "error", "MEDIUM": "warning", "NORMAL": "success"}[self.band]


def classify(percentile: float) -> RiskAssessment:
    """Classify a growth percentile (0–100) into a risk assessment record.

    Thresholds are driven by ``config.HIGH_RISK_MAX_PCT`` and
    ``config.MEDIUM_RISK_MAX_PCT`` so adjustments in one place propagate
    throughout the application and documentation automatically.

    Parameters
    ----------
    percentile:
        Growth percentile in the range [0, 100], as returned by
        ``clinical.percentiles.evaluate``.

    Returns
    -------
    RiskAssessment
        Fully populated risk record ready for rendering.
    """
    colors = config.RISK_COLORS

    if percentile < config.HIGH_RISK_MAX_PCT:
        c = colors["HIGH"]
        return RiskAssessment(
            band="HIGH",
            label="High Risk",
            headline="Below 10th percentile",
            detail=(
                "Screening alert — possible growth restriction (IUGR). "
                "Correlate clinically."
            ),
            color=c["solid"],
            accent=c["accent"],
            soft=c["soft"],
            icon="🔴",
        )

    if percentile < config.MEDIUM_RISK_MAX_PCT:
        c = colors["MEDIUM"]
        return RiskAssessment(
            band="MEDIUM",
            label="Medium Risk",
            headline="10th–25th percentile",
            detail="Borderline growth range — monitor closely on serial scans.",
            color=c["solid"],
            accent=c["accent"],
            soft=c["soft"],
            icon="🟡",
        )

    c = colors["NORMAL"]
    return RiskAssessment(
        band="NORMAL",
        label="Normal",
        headline="Above 25th percentile",
        detail="Within the expected growth curve for gestational age.",
        color=c["solid"],
        accent=c["accent"],
        soft=c["soft"],
        icon="🟢",
    )
