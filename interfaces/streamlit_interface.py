# streamlit_interface.py

import random

import streamlit as st
from PIL import Image

from config.config import FileUploadConfig, model_config_dict
from interfaces.commands import process_command
from services.langfuse_service import handle_langfuse_integration
from services.model_service import ensure_api_key_is_set, instantiate_llm
from utilities.file_utils import save_uploaded_file


def layout_streamlit_ui():
    """Layout the UI elements of the Streamlit application."""
    # Sidebar
    with st.sidebar:
        image = Image.open("images/mole.png")
        st.image(image, width=100)
        display_model_config()
        display_file_and_url_inputs()
        handle_langfuse_integration()
        # Add Clear button
        if st.button("Clear", use_container_width=True):
            clear_session_state()
            st.rerun()
    # Chat history
    display_chat_history()
    # Chat input
    display_chat_widget()


def display_model_config():
    """Configure the left column of the UI for model and input configuration."""
    st.markdown("<h1>m o l e</h1>", unsafe_allow_html=True)
    st.markdown("<h5>Multi-agent Omni LangGraph Executer</h5>", unsafe_allow_html=True)
    with st.expander("Model Configuration", expanded=False):
        model_type = st.selectbox("Select model type:", list(model_config_dict.keys()))
        model_name = st.selectbox(
            "Select model:", list(model_config_dict[model_type].keys())
        )
        selected_model_config = model_config_dict[model_type].get(model_name)
        if not selected_model_config:
            st.error(
                "The selected model configuration was not found. Please select a different model."
            )
            return
        st.session_state.temperature = st.slider(
            "Temperature", 0.0, 1.0, selected_model_config.temperature
        )
        api_key = ensure_api_key_is_set(model_type)
        if not api_key:
            st.warning(
                "API key is required to proceed. Please provide a valid API key."
            )
        st.session_state.recursion_limit = st.number_input(
            "Set Recursion Limit:", min_value=10, max_value=100, value=25
        )
        if api_key:
            st.session_state.llm = instantiate_llm(selected_model_config, api_key)


def display_file_and_url_inputs():
    """Display either file upload or URL input options based on user selection."""
    with st.expander("Information Provider", expanded=False):
        input_type = st.radio("Select input type:", ["File Upload", "URL"])

        try:
            if input_type == "File Upload":
                handle_file_uploads()
                st.session_state.url = None
            else:
                st.session_state.url = handle_url()
                st.session_state.file_upload_config = None
        except Exception as e:
            st.error(f"Error handling {input_type}: {e}")

        if not st.session_state.file_upload_config and not st.session_state.url:
            st.warning("Please provide either a file or URL.")


def handle_file_uploads():
    """Handle file upload input and update the session state for uploaded files."""
    try:
        if "file_uploader_key" not in st.session_state:
            st.session_state.file_uploader_key = "0"

        uploaded_files = st.file_uploader(
            "Upload Files",
            accept_multiple_files=True,
            type=st.session_state.get(
                "allowed_file_types", ["pdf", "csv", "md", "epub", "json", "xml", "txt"]
            ),
            key=st.session_state.file_uploader_key,
        )
        if uploaded_files:
            file_paths = [str(save_uploaded_file(file)) for file in uploaded_files]
            st.session_state.file_upload_config = FileUploadConfig(files=file_paths)
        else:
            st.session_state.file_upload_config = None
    except Exception as e:
        st.error(f"Error uploading files: {e}")


def handle_url() -> str:
    """Handle URL input and return the provided URL."""
    return st.text_input("Enter URL", placeholder="Enter URL")


def display_chat_history():
    """Display the chat history in a scrollable area."""
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def display_chat_widget():
    """Display the chat widget and handle commands."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if prompt := st.chat_input("Enter a command or message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if prompt.startswith("/"):
            process_command(prompt, st.session_state)
        else:
            with st.chat_message("assistant"):
                st.markdown(
                    "I'm sorry, I don't understand that. Please use a command starting with '/'."
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I'm sorry, I don't understand that. Please use a command starting with '/'.",
                    }
                )


def clear_session_state():
    """Clear all session state variables and cache."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()
    # Reset file uploader state
    st.session_state.file_uploader_key = str(random.randint(1000, 9999))
