"""
components.py
=============
Pure HTML builders for the designed parts of the interface. Each returns a
string rendered with ``st.markdown(..., unsafe_allow_html=True)`` so the visual
language is fully controlled rather than the stock widget look.

This module is deliberately free of any Streamlit import so the builders can be
unit-tested headlessly. Decorative emoji are replaced with precise inline SVG
marks (stroke = ``currentColor``) to keep the clinical, non-generic register.
"""

from __future__ import annotations

import config
from clinical.risk import RiskAssessment


# --------------------------------------------------------------------------- #
# Inline SVG icon set (inherit colour via currentColor)
# --------------------------------------------------------------------------- #
ICONS: dict[str, str] = {
    # head-circumference caliper — the product mark
    "caliper": (
        '<svg viewBox="0 0 32 32" fill="none" aria-hidden="true">'
        '<ellipse cx="16" cy="16" rx="11.5" ry="8.5" stroke="currentColor" stroke-width="1.8"/>'
        '<path d="M4.5 16H27.5" stroke="currentColor" stroke-width="1.1" '
        'stroke-dasharray="1.5 2.6" opacity="0.6"/>'
        '<path d="M4.5 12.4V19.6M27.5 12.4V19.6" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round"/>'
        '<circle cx="16" cy="16" r="1.7" fill="currentColor"/></svg>'
    ),
    "warning": (
        '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M12 3.6 21.4 20H2.6z" stroke="currentColor" stroke-width="1.7" '
        'stroke-linejoin="round"/>'
        '<path d="M12 9.6V14" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>'
        '<circle cx="12" cy="17.2" r="1.05" fill="currentColor"/></svg>'
    ),
    "book": (
        '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M12 6.6C10 5.1 6.6 4.8 4 5.5V19c2.6-.7 6-.4 8 1 2-1.4 5.4-1.7 8-1V5.5'
        'c-2.6-.7-6-.4-8 1.1z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>'
        '<path d="M12 6.6V20" stroke="currentColor" stroke-width="1.6"/></svg>'
    ),
    "arrow_right": (
        '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    ),
    "arrow_left": (
        '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M19 12H5M11 6l-6 6 6 6" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    ),
    "scan": (
        '<svg viewBox="0 0 48 48" fill="none" aria-hidden="true">'
        '<path d="M6 15V9a3 3 0 0 1 3-3h6M33 6h6a3 3 0 0 1 3 3v6M42 33v6a3 3 0 0 1-3 3h-6'
        'M15 42H9a3 3 0 0 1-3-3v-6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
        '<ellipse cx="24" cy="24" rx="10" ry="7.5" stroke="currentColor" stroke-width="1.8"/>'
        '<path d="M12 24h24" stroke="currentColor" stroke-width="1.1" '
        'stroke-dasharray="1.6 2.6" opacity="0.6"/></svg>'
    ),
}


def icon(name: str) -> str:
    """Return the inline SVG markup for ``name`` (empty string if unknown)."""
    return ICONS.get(name, "")


# --------------------------------------------------------------------------- #
# Header + safety
# --------------------------------------------------------------------------- #
def header_html() -> str:
    name_main = config.APP_NAME.split("-")[0].strip()  # "Fetal Metrics"
    return f"""
    <div class="fm-header">
      <div class="fm-logo">{ICONS['caliper']}</div>
      <div class="fm-title-wrap">
        <h1>{name_main}<span class="accent">-AI</span></h1>
        <div class="sub">{config.APP_TAGLINE} · automated fetal head-circumference biometry</div>
      </div>
      <div class="spacer"></div>
      <a class="fm-navlink" href="/Methodology" target="_self"
         title="Read the full methodology, references and limitations">
        {ICONS['book']}<span>Methodology</span>
        <span class="arw">{ICONS['arrow_right']}</span>
      </a>
      <span class="fm-pill">Research build · v{config.APP_VERSION}</span>
    </div>
    """


def safety_banner_html() -> str:
    return f"""
    <div class="fm-safety">
      <div class="ico">{ICONS['warning']}</div>
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
# Sidebar brand (paired with the nav in theme.render_sidebar_nav)
# --------------------------------------------------------------------------- #
def sidebar_brand_html() -> str:
    name_main = config.APP_NAME.split("-")[0].strip().replace(" ", "")
    return f"""
    <div class="fm-side-brand">
      <div class="mk">{ICONS['caliper']}</div>
      <div class="nm">{name_main}<span class="d">-AI</span></div>
    </div>
    """


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
    """Risk card. Colour + a status dot carry the meaning (no emoji)."""
    return f"""
    <div class="fm-risk" style="--rk:{risk.color}; --rk-soft:{risk.soft};">
      <div class="top">
        <span class="dot"></span>
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
    return f"""
    <div class="fm-empty">
      <div class="big">{ICONS['scan']}</div>
      <h3>Upload a 2D ultrasound to begin</h3>
      <p>Drop a fetal head ultrasound (PNG/JPG) in the sidebar, set the gestational
      age and choose a segmentation model. The assistant segments the skull, fits an
      ellipse, and reports the calibrated head circumference with its growth percentile.</p>
      <a class="cta" href="/Methodology" target="_self">
        {ICONS['book']}<span>How it works — read the methodology</span>{ICONS['arrow_right']}
      </a>
    </div>
    """
