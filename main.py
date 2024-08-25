# main.py

import os
from typing import Optional

import streamlit as st
from langfuse.callback import CallbackHandler
from pydantic import ValidationError

from modules.app import App
from modules.config.config import (
    AgentConfig,
    FileUploadConfig,
    ModelConfig,
    URLConfig,
    model_config_dict,
)
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


def layout_streamlit_ui():
    """Layout the UI elements of the Streamlit application."""
    left_column, mid_column, right_column = st.columns([0.8, 0.2, 2.0])
    with left_column:
        display_left()
    with mid_column:
        st.image("images/mole.png")
    with right_column:
        display_right()
    display_buttons()


def display_left():
    """Configure the left column of the UI for model and input configuration."""
    st.markdown("<h1>m o l e</h1>", unsafe_allow_html=True)
    st.markdown("<h5>Multi-agent Omni LangGraph Executer</h5>", unsafe_allow_html=True)

    model_type = st.selectbox("Select model type:", list(model_config_dict.keys()))
    model_name = st.selectbox(
        "Select model:", list(model_config_dict[model_type].keys())
    )

    selected_model_config = model_config_dict[model_type][model_name]
    st.session_state.temperature = st.slider(
        "Temperature", 0.0, 1.0, selected_model_config.temperature
    )

    api_key = ensure_api_key_is_set(model_type)

    st.session_state.llm = instantiate_llm(selected_model_config, api_key)

    st.session_state.recursion_limit = st.number_input(
        "Set Recursion Limit:", min_value=10, max_value=100, value=25
    )

    handle_file_uploads()
    st.session_state.url_config = handle_url_input()

    if not st.session_state.file_upload_config and not st.session_state.url_config:
        st.warning("Either a scenario file or scenario URL should be provided!")

    handle_langfuse_integration()


def instantiate_llm(config: ModelConfig, api_key: str):
    try:
        params = {"model": config.model_name, "temperature": config.temperature}
        if config.model_type == "openai":
            params["openai_api_key"] = api_key
            return config.chat_model_class(**params)
        elif config.model_type == "ollama":
            return config.chat_model_class(**params)
        else:
            st.error("Selected model configuration is not supported.")
            return None
    except ValidationError as e:
        st.error("Configuration Error: Check your model parameters and types.")
        st.error(str(e))
        return None


def display_right():
    """Display the JSON configuration editor on the right side of the UI."""
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


def display_buttons():
    """Display buttons for generating and running scenarios."""
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Generate Scenario", use_container_width=True):
                handle_generate_scenario_config()
        with col2:
            if st.button("Run Scenario", use_container_width=True):
                handle_run_scenario_button()


def handle_generate_scenario_config():
    """Generate a scenario configuration based on uploaded files or URL content."""
    if not check_scenario_input():
        return

    documents = load_documents_for_scenario(
        st.session_state.file_upload_config, st.session_state.url_config
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
            url_config=st.session_state.url_config,
        )
        messages = app.setup_and_run_scenario(
            message_placeholder, st.session_state.langfuse_handler
        )
        st.session_state.messages = messages
    except Exception as e:
        st.error(f"Error running the scenario: {str(e)}")


def ensure_api_key_is_set(model_type: str) -> Optional[str]:
    """Ensure that the appropriate API key is set based on the selected model type and return it if needed."""
    api_key_env_vars = {
        "openai": "OPENAI_API_KEY",
    }
    if model_type not in api_key_env_vars:
        return None
    env_var_name = api_key_env_vars[model_type]
    api_key = os.getenv(env_var_name)
    if not api_key:
        api_key = st.text_input(
            f"Enter your API Key for {model_type.capitalize()}:", type="password"
        )
        if api_key:
            os.environ[env_var_name] = (
                api_key  # Set the API key in the environment for future use
            )
    return api_key


def setup_page():
    """Setup the initial Streamlit page configuration."""
    st.set_page_config(layout="wide")
    load_css()


def configure_session_state():
    """Initialize session state variables for file uploads, URL configurations, and others."""
    session_defaults = {
        "file_upload_config": None,
        "url_config": None,
        "llm": None,
        "recursion_limit": None,
        "temperature": None,
        "config_json": None,
        "generated_scenario_config": None,
        "messages": [],
        "langfuse_handler": None,
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)


def check_scenario_input() -> bool:
    """Check if either a file or URL has been provided for scenario generation."""
    if not st.session_state.file_upload_config and not st.session_state.url_config:
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


def handle_file_uploads():
    """Handle file upload input and update the session state for uploaded files."""
    if st.checkbox("Enable File Uploads"):
        with st.expander("File Upload Configuration"):
            uploaded_files = st.file_uploader(
                "Upload Files",
                accept_multiple_files=True,
                type=["pdf", "csv", "md", "epub", "json", "xml", "txt"],
            )
            if uploaded_files:
                file_paths = [str(save_uploaded_file(file)) for file in uploaded_files]
                st.session_state.file_upload_config = FileUploadConfig(files=file_paths)


def handle_url_input() -> Optional[URLConfig]:
    """Handle URL input and return the configuration for the provided URL."""
    if st.checkbox("Enable URL Input", disabled=True):
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


def handle_langfuse_integration() -> Optional[CallbackHandler]:
    """Handle LangFuse integration for LLM operation tracing."""
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
        check_langfuse_connection()
    elif st.checkbox("Enable LangFuse Integration"):
        setup_langfuse_via_ui()


def setup_langfuse_via_ui():
    """Set up LangFuse via UI input and check connection."""
    with st.expander("LangFuse LLM operation tracing"):
        pk = st.text_input("Enter your LangFuse Public Key:", type="password")
        sk = st.text_input("Enter your LangFuse Secret Key:", type="password")
        host = st.text_input("Enter your LangFuse Host Name:")
        if pk and sk and host:
            st.session_state.langfuse_handler = CallbackHandler(
                public_key=pk, secret_key=sk, host=host
            )
            check_langfuse_connection()


def check_langfuse_connection():
    """Check the connection to the LangFuse server."""
    if st.button("Test LangFuse Connection"):
        handler = st.session_state.get("langfuse_handler")
        if handler and handler.auth_check():
            st.success("Authenticated and connected successfully to LangFuse server.")
        else:
            st.error("Failed to authenticate with LangFuse server.")


def load_css():
    """Load custom CSS styles from a file and apply them to the Streamlit application."""
    with open(".css/app_styles.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


if __name__ == "__main__":
    setup_logging()
    setup_page()
    set_api_keys()
    configure_session_state()
    layout_streamlit_ui()
