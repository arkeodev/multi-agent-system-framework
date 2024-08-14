# main.py

import os
from typing import Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from modules.app import App
from modules.config.config import AgentConfig
from modules.utils import (
    load_agent_config,
    load_configuration,
    read_and_format_json,
    set_api_keys,
    setup_logging,
)


def main():
    st.set_page_config(layout="wide")

    # Load CSS styles
    load_css()

    left_column, _, right_column = st.columns([1, 0.1, 2])

    with left_column:
        display_title()
        model_choice, recursion_limit = display_ui()

    with right_column:
        config_json = display_scenario()

    user_config = (
        validate_and_parse_json(config_json) if config_json else load_agent_config()
    )

    message_placeholder = st.empty()  # Placeholder for dynamic message updates

    if st.button("Run Scenario"):
        if user_config:
            try:
                llm = ChatOpenAI(model=model_choice)
                app = App(
                    llm=llm,
                    recursion_limit=recursion_limit,
                    config=user_config.model_dump(),
                )
                messages = app.setup_and_run_scenario(
                    recursion_limit, message_placeholder
                )  # Pass the placeholder
            except Exception as e:
                st.error(f"Error running the scenario: {str(e)}")
        else:
            st.error("Configuration is required to run the scenario.")


def display_title() -> None:
    """Display title and image."""
    # Use a container to wrap the image for specific styling
    st.markdown("<h1>m o l e</h1>", unsafe_allow_html=True)
    st.image("images/mole.png")
    st.markdown("<h3>Multi-agent Omni Langraph Executer</h3>", unsafe_allow_html=True)


def display_ui() -> tuple:
    """Displays the user interface for model selection and API key input."""
    config = load_configuration()

    if not os.path.exists(".env"):
        api_key = st.text_input(
            label="Enter your OpenAI API Key:",
            type="password",
            placeholder="sk-------------",
        )
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        set_api_keys()

    model_choice = st.selectbox("Select the model to use:", config["models"], index=0)
    recursion_limit = st.number_input(
        "Set Recursion Limit:", min_value=10, max_value=100, value=25
    )

    return model_choice, recursion_limit


def display_scenario() -> str:
    """Inputs JSON config input with formatted placeholder from agent_config.json.
    Returns the inputted JSON if provided; otherwise, returns the placeholder JSON."""
    placeholder_json = read_and_format_json("./modules/config/agent_config.json")
    config_json = st.text_area(
        "Configuration JSON (optional)",
        value="",
        placeholder=placeholder_json,
        height=500,
    )

    # Check if the text area is empty (taking into account whitespace)
    if not config_json.strip():
        return placeholder_json
    else:
        return config_json


def validate_and_parse_json(json_input: str) -> Optional[AgentConfig]:
    """Validate and parse JSON configuration using Pydantic."""
    try:
        return AgentConfig.model_validate_json(json_input)
    except ValidationError as e:
        st.error(f"Invalid configuration: {e.json()}")
        return None


def load_css():
    """Load CSS styles from a css file."""
    with open(".css/app_styles.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


if __name__ == "__main__":
    setup_logging()
    main()
