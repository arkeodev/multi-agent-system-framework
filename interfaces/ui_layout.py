import logging

import streamlit as st
from PIL import Image

from config.config import model_config_dict
from core.state_manager import ChatState
from services.langfuse_service import handle_langfuse_integration
from services.model_service import ensure_api_key_is_set, instantiate_llm


def layout_streamlit_ui():
    """Layout the UI elements of the Streamlit application."""
    logging.info("Setting up Streamlit UI layout")
    configure_session_state()

    # Sidebar
    with st.sidebar:
        image = Image.open("images/mole.png")
        st.image(image, width=100)
        display_model_config()
        handle_langfuse_integration()
        if st.button("Clear", use_container_width=True):
            logging.info("Clear button pressed")
            clear_session_state()
            st.rerun()


def display_chat_widget():
    logging.info("Setting up chat widget")
    if prompt := st.chat_input("Enter a query, command, or input"):
        logging.info(f"User input received: {prompt}")
        return prompt
    return None


def display_model_config():
    """Configure the left column of the UI for model and input configuration."""
    logging.info("Displaying model configuration")
    st.markdown("<h1>m o l e</h1>", unsafe_allow_html=True)
    st.markdown("<h5>Multi-agent Omni LangGraph Executer</h5>", unsafe_allow_html=True)
    with st.expander("Model Configuration", expanded=False):
        model_type = st.selectbox(
            "Select model type:",
            list(model_config_dict.keys()),
            key="model_type_selectbox",
        )
        model_name = st.selectbox(
            "Select model:",
            list(model_config_dict[model_type].keys()),
            key="model_name_selectbox",
        )
        selected_model_config = model_config_dict[model_type].get(model_name)
        if not selected_model_config:
            st.error(
                "The selected model configuration was not found. Please select a different model."
            )
            logging.error(f"Invalid model configuration: {model_type} - {model_name}")
            return
        st.session_state.temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            selected_model_config.temperature,
            key="temperature_slider",
        )
        api_key = ensure_api_key_is_set(model_type)
        if not api_key:
            st.warning(
                "API key is required to proceed. Please provide a valid API key."
            )
            logging.warning(f"API key not set for model type: {model_type}")
        st.session_state.recursion_limit = st.number_input(
            "Set Recursion Limit:", min_value=10, max_value=100, value=25
        )
        if api_key:
            st.session_state.llm = instantiate_llm(selected_model_config, api_key)
            logging.info(f"LLM instantiated for model: {model_name}")


def configure_session_state():
    if "step" not in st.session_state:
        st.session_state.step = "query_specification"
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = ChatState.WAITING_FOR_QUERY
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
