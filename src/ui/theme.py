"""
theme.py
========
Page configuration, font loading, stylesheet injection and the shared sidebar
navigation. Keeping the chrome here means ``app.py`` and the methodology page
stay about content, not plumbing.

Design register: clinical-light "medical instrument" — Inter for the interface,
IBM Plex Mono for numeric readouts, Source Serif 4 for the long methodology
document. The full visual system lives in ``assets/styles.css``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from ui import components as C

# Interface (Inter) + numeric readouts (IBM Plex Mono) + long-form docs
# (Source Serif 4). Loaded once per page via ``inject_styles``.
_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Inter:wght@400;500;600;700&"
    "family=IBM+Plex+Mono:wght@400;500;600&"
    "family=Source+Serif+4:ital,wght@0,400;0,500;0,600;0,700;1,400&"
    'display=swap" rel="stylesheet">'
)


def configure_page(active: str = "analyze") -> None:
    """Set Streamlit page config. Call once, first thing on every page."""
    st.set_page_config(
        page_title=f"{config.APP_NAME} — {config.APP_TAGLINE}",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data
def _load_stylesheet(css_path_str: str) -> str:
    path = Path(css_path_str)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def inject_styles() -> None:
    """Inject fonts + the project stylesheet into the current page."""
    st.markdown(_FONTS, unsafe_allow_html=True)
    css_str = _load_stylesheet(str(config.ASSETS_DIR / "styles.css"))
    if css_str:
        st.markdown(f"<style>{css_str}</style>", unsafe_allow_html=True)


def render_sidebar_nav(active: str = "analyze") -> None:
    """Render the shared brand lockup + controlled page navigation.

    ``active`` is "analyze" or "methodology" and only affects which nav item is
    visually current (Streamlit also sets ``aria-current`` automatically).
    """
    with st.sidebar:
        st.markdown(C.sidebar_brand_html(), unsafe_allow_html=True)
        st.markdown('<div class="fm-nav-hint">Navigate</div>', unsafe_allow_html=True)
        st.page_link("app.py", label="Analyze", icon=":material/monitor_heart:")
        st.page_link(
            "pages/1_Methodology.py",
            label="Methodology",
            icon=":material/menu_book:",
        )
        st.divider()
