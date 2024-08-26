import streamlit as st

from utilities.utils import set_api_keys, setup_logging


def setup_page():
    """Setup the initial Streamlit page configuration."""
    st.set_page_config(layout="wide")
    load_css()


def configure_session_state():
    """Initialize session state variables for file uploads, URL configurations, and others."""
    session_defaults = {
        "file_upload_config": None,
        "url": "",
        "llm": None,
        "recursion_limit": None,
        "temperature": None,
        "config_json": None,
        "generated_scenario_config": None,
        "messages": [],
        "langfuse_handler": None,
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)


def load_css():
    """Load custom CSS styles from a file and apply them to the Streamlit application."""
    with open(".css/app_styles.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
