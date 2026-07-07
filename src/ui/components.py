"""
components.py
=============
Small, pure HTML builders for the designed parts of the interface. Each returns
a string to be rendered with ``st.markdown(..., unsafe_allow_html=True)`` so the
visual language is fully controlled (not the stock widget look).
"""

from __future__ import annotations

import config
from clinical.risk import RiskAssessment


# --------------------------------------------------------------------------- #
# Header + safety
# --------------------------------------------------------------------------- #
def header_html() -> str:
    return f"""
    <div class="fm-header">
      <div class="fm-title">
        <div class="fm-logo">🩺</div>
        <div>
          <h1>{config.APP_NAME.split('-')[0]}<span class="accent">-AI</span></h1>
          <div class="sub">{config.APP_TAGLINE} · automated fetal head-circumference biometry</div>
        </div>
        <span class="fm-pill">Research build v{config.APP_VERSION}</span>
      </div>
    </div>
    """


def safety_banner_html() -> str:
    return f"""
    <div class="fm-safety">
      <div class="ico">⚠️</div>
      <div class="txt"><b>Research prototype — not for clinical diagnosis.</b><br>
        {config.SAFETY_NOTICE}</div>
    </div>
    """


def section_label(text: str) -> str:
    return f'<div class="fm-section">{text}</div>'


def image_caption(title: str, right: str = "") -> str:
    r = f'<span class="dot">{right}</span>' if right else ""
    return f'<div class="fm-imgcap"><span>{title}</span>{r}</div>'


# --------------------------------------------------------------------------- #
# Dashboard metric cards
# --------------------------------------------------------------------------- #
def metric_card_html(label: str, value: str, unit: str = "", sub_html: str = "") -> str:
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    sub = f'<div class="sub">{sub_html}</div>' if sub_html else ""
    return f"""
    <div class="fm-metric">
      <div class="label">{label}</div>
      <div class="value">{value}{unit_html}</div>
      {sub}
    </div>
    """


def risk_card_html(risk: RiskAssessment) -> str:
    return f"""
    <div class="fm-risk" style="--rk:{risk.color}; --rk-soft:{risk.soft};">
      <div class="top">
        <span>{risk.icon}</span>
        <span class="badge">{risk.label}</span>
      </div>
      <div class="headline">{risk.headline}</div>
      <div class="detail">{risk.detail}</div>
    </div>
    """


# --------------------------------------------------------------------------- #
# Percentile gauge
# --------------------------------------------------------------------------- #
def gauge_html(percentile: float) -> str:
    """Zoned distribution bar with a marker at the measured percentile."""
    pos = max(0.8, min(99.2, percentile))  # keep marker visible at extremes

    def tick(p: float, label: str) -> str:
        return (f'<div class="tick" style="left:{p}%;">'
                f'<span class="bar"></span>{label}</div>')

    ticks = "".join([
        tick(10, "10th"), tick(25, "25th"), tick(50, "50th"), tick(90, "90th"),
    ])

    return f"""
    <div class="fm-gauge">
      <div class="track">
        <div class="bubble" style="left:{pos}%;">{percentile:.1f}th</div>
        <div class="marker" style="left:{pos}%;"></div>
      </div>
      <div class="ticks">{ticks}</div>
      <div class="fm-legend">
        <div class="item"><span class="sw" style="background:{config.RISK_COLORS['HIGH']['solid']}"></span>High risk &lt;10th</div>
        <div class="item"><span class="sw" style="background:{config.RISK_COLORS['MEDIUM']['solid']}"></span>Borderline 10–25th</div>
        <div class="item"><span class="sw" style="background:{config.RISK_COLORS['NORMAL']['solid']}"></span>Normal &gt;25th</div>
      </div>
    </div>
    """


# --------------------------------------------------------------------------- #
# Timing / provenance strip
# --------------------------------------------------------------------------- #
def timing_strip_html(items: list[tuple[str, str]]) -> str:
    parts = []
    for i, (k, v) in enumerate(items):
        if i:
            parts.append('<span class="sep">·</span>')
        parts.append(f'<span><span class="k">{k}</span> <span class="v">{v}</span></span>')
    return f'<div class="fm-strip">{"".join(parts)}</div>'


# --------------------------------------------------------------------------- #
# Empty state
# --------------------------------------------------------------------------- #
def empty_state_html() -> str:
    return """
    <div class="fm-empty">
      <div class="big">🖼️</div>
      <h3>Upload a 2D ultrasound to begin</h3>
      <p>Drop a fetal head ultrasound (PNG/JPG) in the sidebar, set the gestational
      age and choose a model. The assistant will segment the skull, fit an ellipse,
      and report the calibrated head circumference with its growth percentile.</p>
    </div>
    """
