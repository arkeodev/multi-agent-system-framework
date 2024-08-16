# main.py

import logging
import os
from typing import Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from modules.app import App
from modules.config.config import AgentConfig, FileUploadConfig, URLConfig
from modules.utils import (
    load_agent_config,
    read_and_format_json,
    save_uploaded_file,
    set_api_keys,
    setup_logging,
)


def main():
    """Main function to set up the Streamlit application layout and handle user interactions."""
    st.set_page_config(layout="wide")
    load_css()

    left_column, _, right_column = st.columns([1.1, 0.1, 1.9])

    st.session_state.file_upload_config, st.session_state.url_config = None, None

    with left_column:
        display_title()
        model_choice, recursion_limit = display_ui()
        st.session_state.file_upload_config = handle_file_uploads()
        logging.info(f"Uploaded files are: {st.session_state.file_upload_config}")
        st.session_state.url_config = handle_url_input()
        if (
            st.session_state.file_upload_config is None
            and st.session_state.url_config is None
        ):
            st.warning("Either a scenario file or scenario URL should be provided!")

    with right_column:
        agent_config = display_scenario()

    message_placeholder = st.empty()  # Placeholder for streaming scenario results

    if st.button("Run Scenario", help="Click to run the scenario"):
        run_scenario(
            model_choice,
            recursion_limit,
            agent_config,
            st.session_state.file_upload_config,
            st.session_state.url_config,
            message_placeholder,
        )


def run_scenario(
    model_choice: str,
    recursion_limit: int,
    agent_config: str,
    file_upload_config: Optional[FileUploadConfig],
    url_config: Optional[URLConfig],
    message_placeholder,
):
    """Execute the multi-agent scenario using the provided configurations."""
    user_config = (
        validate_and_parse_json(agent_config) if agent_config else load_agent_config()
    )

    if user_config:
        try:
            llm = ChatOpenAI(model=model_choice)
            app = App(
                llm=llm,
                recursion_limit=recursion_limit,
                agent_config=user_config.model_dump(),
                file_config=file_upload_config,
                url_config=url_config,
            )
            app.setup_and_run_scenario(recursion_limit, message_placeholder)
        except Exception as e:
            st.error(f"Error running the scenario: {str(e)}")
    else:
        st.error("Configuration is required to run the scenario.")


def display_title():
    """Displays the application title and logo."""
    st.markdown("<h1>m o l e</h1>", unsafe_allow_html=True)
    st.image("images/mole.png")
    st.markdown("<h3>Multi-agent Omni Langraph Executer</h3>", unsafe_allow_html=True)


def display_ui() -> tuple:
    """Displays the UI elements for model selection and recursion limit input."""
    if not os.path.exists(".env"):
        api_key = st.text_input(
            label="Enter your OpenAI API Key:",
            type="password",
            placeholder="sk-------------",
        )
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        set_api_keys()

    model_choice = st.selectbox(
        "Select the model to use:", ["gpt-4o-mini", "gpt-4o"], index=0
    )
    recursion_limit = st.number_input(
        "Set Recursion Limit:", min_value=10, max_value=100, value=25
    )

    return model_choice, recursion_limit


def handle_file_uploads() -> Optional[FileUploadConfig]:
    """Handles file upload input and returns the configuration for the uploaded files."""
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
                return FileUploadConfig(files=file_paths)
    return None


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


def display_scenario() -> str:
    """Displays the text area for entering or editing the JSON configuration."""
    placeholder_json = read_and_format_json("./modules/config/agent_config.json")
    config_json = st.text_area(
        "Configuration JSON (optional)",
        value="",
        placeholder=placeholder_json,
        height=500,
    )

    return config_json if config_json.strip() else placeholder_json


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
