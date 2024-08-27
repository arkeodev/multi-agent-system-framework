# streamlit_interfice.py

import streamlit as st

from config.config import SAMPLE_AGENT_CONFIG, FileUploadConfig, model_config_dict
from services.config_service import generate_config, run_config, visualize_graph
from services.langfuse_service import handle_langfuse_integration
from services.model_service import ensure_api_key_is_set, instantiate_llm
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
            st.error("The selected model configuration was not found. Please select a different model.")
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
        else:
            st.session_state.file_upload_config = None
    except Exception as e:
        st.error(f"Error uploading files: {e}")


def handle_url() -> str:
    """Handle URL input and return the provided URL."""
    return st.text_input("Enter URL", placeholder="Enter URL")


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


def display_buttons():
    """Display buttons for generating, running, and visualizing agents in graph."""
    with st.container():
        col1, col2 = st.columns([1, 2])

        with col1:
            col1a, col1b = st.columns([1, 1])
            with col1a:
                if st.button("Generate Agent Configuration", use_container_width=True):
                    generate_config()
            with col1b:
                if st.button("Visualize Graph", use_container_width=True):
                    visualize_graph()

        with col2:
            if st.button("Run", use_container_width=True):
                run_config()
