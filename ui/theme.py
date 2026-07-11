from pathlib import Path
import streamlit as st


def load_theme() -> None:
    """
    Load the global CSS file.
    """

    css_file = Path("assets/css/style.css")

    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as file:
            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True,
            )