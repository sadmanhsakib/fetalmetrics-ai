"""
theme.py
========
Page configuration + one-time injection of Google Fonts and the project
stylesheet. Keeping this here means ``app.py`` stays about layout, not chrome.
"""

from __future__ import annotations

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


def inject_styles() -> None:
    """Inject fonts + the stylesheet into the page."""
    st.markdown(_FONTS, unsafe_allow_html=True)
    css_path = config.ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)
