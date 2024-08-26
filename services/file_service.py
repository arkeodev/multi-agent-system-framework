# file_service.py

import streamlit as st

from config.config import FileUploadConfig
from utilities.file_utils import save_uploaded_file


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
