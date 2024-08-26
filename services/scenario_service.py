# scenario_service.py

import streamlit as st

from agents.scenario_generator import (
    generate_scenario_config,
    load_documents_for_scenario,
)
from config.config import AgentConfig
from core.app import App


def display_buttons():
    """Display buttons for generating and running scenarios."""
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Generate Agent Configuration", use_container_width=True):
                handle_generate_scenario_config()
        with col2:
            if st.button("Run Scenario", use_container_width=True):
                handle_run_scenario_button()


def handle_generate_scenario_config():
    """Generate a scenario configuration based on uploaded files or URL content."""
    if not check_scenario_input():
        return
    documents = load_documents_for_scenario(
        st.session_state.file_upload_config, st.session_state.url
    )
    generated_config = generate_scenario_config(st.session_state.llm, documents)
    st.session_state.config_json = generated_config
    st.rerun()


def handle_run_scenario_button():
    """Handle scenario execution."""
    if not check_scenario_input() or not check_scenario_config():
        return
    st.session_state.messages = []
    message_placeholder = st.empty()
    try:
        user_config = AgentConfig.model_validate_json(st.session_state.config_json)
        app = App(
            llm=st.session_state.llm,
            recursion_limit=st.session_state.recursion_limit,
            agent_config=user_config.model_dump(),
            file_config=st.session_state.file_upload_config,
            url=st.session_state.url,
        )
        messages = app.setup_and_run_scenario(
            message_placeholder, st.session_state.langfuse_handler
        )
        st.session_state.messages = messages
    except Exception as e:
        st.error(f"Error running the scenario: {str(e)}")


def check_scenario_input() -> bool:
    """Check if either a file or URL has been provided for scenario generation."""
    if not st.session_state.file_upload_config and not st.session_state.url:
        st.warning(
            "Please upload a file or provide a URL to generate the scenario config."
        )
        return False
    return True


def check_scenario_config() -> bool:
    """Check if a scenario configuration exists."""
    if not st.session_state.config_json:
        st.warning(
            "Please provide a configuration file either manually or using the generate button."
        )
        return False
    return True