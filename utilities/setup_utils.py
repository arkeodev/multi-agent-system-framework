# setup_utils.py

import logging
import os

import streamlit as st
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler


def setup_logging():
    """Sets up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


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


def set_api_keys(env_file_path=".env"):
    """Loads and sets necessary API keys for OpenAI and LangFuse."""
    logging.info("Attempting to load API keys from specified .env file.")
    load_dotenv(env_file_path)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logging.info("OpenAI API key loaded successfully.")

    langfuse_keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    if all(os.getenv(key) for key in langfuse_keys):
        logging.info("LangFuse API keys loaded successfully.")


def setup_langfuse_keys(pk: str, sk: str, host: str) -> CallbackHandler:
    """Creates a LangFuse CallbackHandler with the provided keys."""
    return CallbackHandler(public_key=pk, secret_key=sk, host=host)
