# config_service.py

import logging

import streamlit as st

from agents.agent_config import generate_config_json
from agents.rag import get_documents
from config.config import AgentConfig
from core.app import App


def generate_config():
    """Generate a configuration based on uploaded files or URL content."""
    if not check_input():
        return
    documents = get_documents(
        getattr(st.session_state.file_upload_config, "files", None),
        st.session_state.url,
    )
    generated_config = generate_config_json(st.session_state.llm, documents)
    if not generated_config:
        st.error("Failed to generate configuration.")
        return
    st.session_state.config_json = generated_config
    st.rerun()


def run_config():
    """Handle execution."""
    if not check_input() or not verify_config():
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
            langfuse_handler=st.session_state.langfuse_handler,
        )
        messages = app.execute_graph(message_placeholder)  # Pass the placeholder here
        st.session_state.messages = messages
    except Exception as e:
        st.error(f"Error running the config: {str(e)}")


def visualize_graph():
    """Handle graph visualization."""
    if not verify_config():
        return
    try:
        user_config = AgentConfig.model_validate_json(st.session_state.config_json)
        logging.info(f"Visualizing graph for config: {user_config}")
        app = App(
            llm=st.session_state.llm,
            recursion_limit=st.session_state.recursion_limit,
            agent_config=user_config.model_dump(),
            file_config=st.session_state.file_upload_config,
            url=st.session_state.url,
            langfuse_handler=st.session_state.langfuse_handler,
        )
        graph_image = app.visualise_graph()
        st.image(graph_image, caption="Graph Visualization")
    except Exception as e:
        st.error(f"Error visualizing the graph: {str(e)}")


def check_input() -> bool:
    """Check if either a file or URL has been provided for config generation."""
    if not st.session_state.file_upload_config and not st.session_state.url:
        st.warning("Please upload a file or provide a URL to generate the config file.")
        return False
    return True


def verify_config() -> bool:
    """Verify the configuration exists."""
    if not st.session_state.config_json:
        st.warning(
            "Please provide a configuration file either manually or using the generate button."
        )
        return False
    return True
