# streamlit_interfice.py

import streamlit as st

from config.config import SAMPLE_AGENT_CONFIG, FileUploadConfig
from services.langfuse_service import handle_langfuse_integration
from services.model_service import display_model_config
from services.scenario_service import display_buttons
from utilities.file_utils import save_uploaded_file


def layout_streamlit_ui():
    """Layout the UI elements of the Streamlit application."""
    left_column, mid_column, right_column = st.columns([0.8, 0.2, 2.0])
    with left_column:
        try:
            display_model_config()
            display_file_and_url_inputs()
            handle_langfuse_integration()
        except Exception as e:
            st.error(f"Error displaying components: {e}")
    with mid_column:
        st.image("images/mole.png")
    with right_column:
        display_agent_config()
    display_buttons()


def display_file_and_url_inputs():
    """Display the file upload and URL input options."""
    with st.expander("Information Provider", expanded=False):
        try:
            handle_file_uploads()
            st.session_state.url = handle_url()
        except Exception as e:
            st.error(f"Error handling file uploads or URL input: {e}")
        if not st.session_state.file_upload_config and not st.session_state.url:
            st.warning("Either a scenario file or scenario URL should be provided!")


def display_agent_config():
    """Display the JSON configuration editor on the right side of the UI."""
    from utilities.json_utils import format_json, read_json

    try:
        placeholder_json = format_json(read_json(SAMPLE_AGENT_CONFIG))
    except Exception as e:
        st.error(f"Error loading or formatting JSON: {e}")
    st.markdown("<h5>Agent Configuration</h5>", unsafe_allow_html=True)
    st.markdown("Edit the configuration if not applicable.")
    st.text_area(
        label=" ",
        label_visibility="hidden",
        value=st.session_state.config_json or "",
        placeholder=placeholder_json,
        key="generated_config_text_area",
        height=500,
    )


def handle_file_uploads():
    """Handle file upload input and update the session state for uploaded files."""
    try:
        uploaded_files = st.file_uploader(
            "Upload Files",
            accept_multiple_files=True,
            type=st.session_state.get(
                "allowed_file_types", ["pdf", "csv", "md", "epub", "json", "xml", "txt"]
            ),
        )
        if uploaded_files:
            file_paths = [str(save_uploaded_file(file)) for file in uploaded_files]
            st.session_state.file_upload_config = FileUploadConfig(files=file_paths)
    except Exception as e:
        st.error(f"Error uploading files: {e}")


def handle_url() -> str:
    """Handle URL input and return the configuration for the provided URL."""
    return st.text_input("Enter URL", placeholder="Enter URL")
