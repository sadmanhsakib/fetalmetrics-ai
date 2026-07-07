"""
theme.py
========
Page configuration + one-time injection of Google Fonts and the project
stylesheet. Keeping this here means ``app.py`` stays about layout, not chrome.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import config

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Sans:wght@400;500;600;700&'
    'family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
)


def configure_page() -> None:
    """Set Streamlit page config. Call once, first thing in app.py."""
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
    """Inject fonts + the stylesheet into the page."""
    st.markdown(_FONTS, unsafe_allow_html=True)
    css_str = _load_stylesheet(str(config.ASSETS_DIR / "styles.css"))
    if css_str:
        st.markdown(f"<style>{css_str}</style>", unsafe_allow_html=True)
