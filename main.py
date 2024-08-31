import logging

import streamlit as st

from config.model_config import model_config_dict
from core.event_manager import EventManager
from core.state_manager import StateManager
from interfaces.ui_layout import display_chat_widget, layout_streamlit_ui
from utilities.setup_utils import (
    configure_session_state,
    set_api_keys,
    setup_logging,
    setup_page,
)


def main():
    setup_logging()
    setup_page()
    set_api_keys()
    configure_session_state()
    # Layout the UI
    layout_streamlit_ui()

    # Initialize EventManager and StateManager
    if "event_emitter" not in st.session_state:
        st.session_state.event_emitter = EventManager(st.session_state.llm)
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = StateManager(st.session_state.event_emitter)

    # Display existing chat messages
    conversation_history = st.session_state.state_manager.get_conversation_history()
    for message in conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    user_input = display_chat_widget()
    if user_input:
        response = st.session_state.state_manager.handle_user_input(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
