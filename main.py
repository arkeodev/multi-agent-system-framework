# main.py

import streamlit as st

from interfaces.streamlit_interface import layout_streamlit_ui
from utilities.setup_utils import (
    configure_session_state,
    set_api_keys,
    setup_logging,
    setup_page,
)

if __name__ == "__main__":
    setup_logging()
    setup_page()
    set_api_keys()
    configure_session_state()
    layout_streamlit_ui()
