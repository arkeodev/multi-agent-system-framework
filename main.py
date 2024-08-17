# main.py

import os
from typing import Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from modules.app import App
from modules.config.config import AgentConfig, FileUploadConfig, URLConfig
from modules.scenario_generator import (
    generate_scenario_config,
    load_documents_for_scenario,
)
from modules.utils import (
    format_json,
    read_json,
    save_uploaded_file,
    set_api_keys,
    setup_logging,
)


def main():
    """Main function to set up the Streamlit application layout and handle user interactions."""
    setup_page()
    set_api_keys()
    configure_session_state()

    left_column, mid_column, right_column = st.columns([0.8, 0.2, 2.0])

    with left_column:
        display_left()

    with mid_column:
        st.image("images/mole.png")

    with right_column:
        display_right()

    # Create a full-width container for the buttons
    st.markdown("---")  # Add a horizontal line to separate sections
    with st.container():
        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("Generate Scenario", use_container_width=True):
                handle_generate_scenario_config()

        with col2:
            if st.button("Run Scenario", use_container_width=True):
                handle_run_scenario_button()


def handle_run_scenario_button():
    """Handles the action for the 'Run Scenario' button."""
    if not check_scenario_input():
        return
    elif not check_scenario_config():
        return

    st.session_state.messages = []  # Clear any previous messages
    message_placeholder = st.empty()
    user_config = validate_and_parse_json(st.session_state.config_json)

    if not user_config:
        st.error("Configuration is required to run the scenario.")
        return

    try:
        app = App(
            llm=st.session_state.llm,
            recursion_limit=st.session_state.recursion_limit,
            agent_config=user_config.model_dump(),
            file_config=st.session_state.file_upload_config,
            url_config=st.session_state.url_config,
        )

        # Collect messages from the scenario
        messages = app.setup_and_run_scenario(
            st.session_state.recursion_limit, message_placeholder
        )
        st.session_state.messages = messages
    except Exception as e:
        st.error(f"Error running the scenario: {str(e)}")


def setup_page():
    """Sets up the Streamlit page configuration and loads custom CSS."""
    st.set_page_config(layout="wide")
    load_css()


def configure_session_state():
    """Initializes session state variables for file uploads, URL configurations, and others."""
    session_defaults = {
        "file_upload_config": None,
        "url_config": None,
        "llm": None,
        "recursion_limit": None,
        "model_choice": None,
        "temperature": None,
        "config_json": None,
        "generated_scenario_config": None,
        "messages": [],
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_scenario_input() -> bool:
    """Checks if either a file or URL has been provided for scenario generation."""
    if not st.session_state.file_upload_config and not st.session_state.url_config:
        st.warning(
            "Please upload a file or provide a URL to generate the scenario config."
        )
        return False
    return True


def check_scenario_config() -> bool:
    """Checks if a scenario configuration exist."""
    if not st.session_state.config_json:
        st.warning(
            "Please provide a configuration file either manually or using the generate button."
        )
        return False
    return True


def handle_generate_scenario_config():
    """Generates a scenario configuration based on uploaded files or URL content."""
    if not check_scenario_input():
        return

    documents = load_documents_for_scenario(
        st.session_state.file_upload_config, st.session_state.url_config
    )
    generated_config = generate_scenario_config(st.session_state.llm, documents)

    st.session_state.config_json = generated_config
    st.rerun()


def display_left():
    """Displays the UI elements for model selection, recursion limit input, file uploads, and URL input."""
    st.markdown("<h1>m o l e</h1>", unsafe_allow_html=True)
    st.markdown("<h5>Multi-agent Omni LangGraph Executer</h5>", unsafe_allow_html=True)

    st.session_state.model_choice = st.selectbox(
        "Select the model to use:", ["gpt-4o-mini", "gpt-4o"], index=0
    )

    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3)

    # Ensure API key is set and create ChatOpenAI object
    api_key = ensure_api_key_is_set()
    if api_key:
        st.session_state.llm = ChatOpenAI(
            model=st.session_state.model_choice,
            openai_api_key=api_key,
            temperature=st.session_state.temperature,
        )
    else:
        st.warning("API key is required to proceed.")

    st.session_state.recursion_limit = st.number_input(
        "Set Recursion Limit:", min_value=10, max_value=100, value=25
    )

    handle_file_uploads()
    st.session_state.url_config = handle_url_input()

    if not st.session_state.file_upload_config and not st.session_state.url_config:
        st.warning("Either a scenario file or scenario URL should be provided!")


def ensure_api_key_is_set():
    """Ensures the API key is set either by user input or from an environment file."""
    if not os.getenv(
        "OPENAI_API_KEY"
    ):  # Check if the key is already in the environment
        api_key = st.text_input(
            label="Enter your OpenAI API Key:",
            type="password",
            placeholder="sk-------------",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key  # Set the key in the environment
    else:
        api_key = os.getenv("OPENAI_API_KEY")
    return api_key


def handle_file_uploads():
    """Handles file upload input and updates the session state for uploaded files."""
    enable_files_input = st.checkbox("Enable File Uploads")
    if enable_files_input:
        with st.expander("File Upload Configuration"):
            uploaded_files = st.file_uploader(
                "Upload Files",
                accept_multiple_files=True,
                type=[
                    "pdf",
                    "csv",
                    "md",
                    "epub",
                    "json",
                    "xml",
                    "txt",
                ],
            )
            if uploaded_files:
                file_paths = [str(save_uploaded_file(file)) for file in uploaded_files]
                st.session_state.file_upload_config = FileUploadConfig(files=file_paths)


def handle_url_input() -> Optional[URLConfig]:
    """Handles URL input and returns the configuration for the provided URL."""
    enable_url_input = st.checkbox("Enable URL Input", disabled=True)
    if enable_url_input:
        with st.expander("URL Configuration"):
            url = st.text_input("Enter URL")
            if url:
                exclusion_pattern = st.text_input("Enter patterns to exclude")
                max_depth = st.number_input(
                    "Set Max Scraping Depth", min_value=1, max_value=3, value=2
                )
                return URLConfig(
                    url=url, exclusion_pattern=exclusion_pattern, max_depth=max_depth
                )
    return None


def display_right():
    """Displays the text area for entering or editing the JSON configuration."""
    placeholder_json = format_json(
        read_json("./modules/config/sample_agent_config.json")
    )

    st.text_area(
        "Configuration JSON",
        value=st.session_state.config_json or "",
        placeholder=placeholder_json,
        key="generated_config_text_area",
        height=500,
    )


def validate_and_parse_json(json_input: str) -> Optional[AgentConfig]:
    """Validates and parses the JSON configuration using Pydantic."""
    try:
        return AgentConfig.model_validate_json(json_input)
    except ValidationError as e:
        st.error(f"Invalid configuration: {e.json()}")
        return None


def load_css():
    """Loads custom CSS styles from a file and applies them to the Streamlit application."""
    with open(".css/app_styles.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


if __name__ == "__main__":
    setup_logging()
    main()
