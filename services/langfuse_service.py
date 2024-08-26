# langfuse_service.py

import os

import streamlit as st
from langfuse.callback import CallbackHandler


def handle_langfuse_integration():
    """Handle LangFuse integration for LLM operation tracing."""
    with st.expander("LangFuse Integration", expanded=False):
        if all(
            os.getenv(key)
            for key in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
        ):
            st.warning("LangFuse enabled for LLM operation tracing.")
            st.session_state.langfuse_handler = CallbackHandler(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST"),
            )
        else:
            setup_langfuse_via_ui()
        check_langfuse_connection()


def setup_langfuse_via_ui():
    """Set up LangFuse via UI input and check connection."""
    pk = st.text_input("Enter your LangFuse Public Key:", type="password")
    sk = st.text_input("Enter your LangFuse Secret Key:", type="password")
    host = st.text_input("Enter your LangFuse Host Name:")
    if pk and sk and host:
        st.session_state.langfuse_handler = CallbackHandler(
            public_key=pk, secret_key=sk, host=host
        )


def check_langfuse_connection():
    """Check the connection to the LangFuse server."""
    if st.button("Test LangFuse Connection"):
        handler = st.session_state.get("langfuse_handler")
        if handler and handler.auth_check():
            st.success("Authenticated and connected successfully to LangFuse server.")
        else:
            st.error("Failed to authenticate with LangFuse server.")
