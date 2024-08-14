# main.py

import os

import pydantic
import streamlit as st
from langchain_openai import ChatOpenAI

from modules.app import App
from modules.utils import (load_agent_config, load_configuration, set_api_keys,
                           setup_logging)


def main():
    """Main function to initialize and run the Streamlit UI."""
    st.set_page_config(layout="wide")
    st.title("Multi-Agent Scenario Executer")

    # UI Setup
    left_column, _, right_column = st.columns([1, 0.1, 2.6])

    with left_column:
        model_choice, recursion_limit = display_ui()

    with right_column:
        scenario = display_scenario()

    # Trigger scenario execution
    if st.button("Run Scenario"):
        try:
            # logging.info(f"API Key: {api_key}")
            llm = ChatOpenAI(model=model_choice)
            app = App(llm, recursion_limit)
            messages = app.setup_and_run_scenario(scenario, recursion_limit)
            st.write("\n".join(messages))
        except pydantic.v1.error_wrappers.ValidationError as e:
            st.error(f"Please enter a valid API key: {str(e)}")


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
    """Displays the scenario text input area."""
    agent_config = load_agent_config()
    scenario = st.text_area("Scenario Text", agent_config["scenario"], height=300)
    return scenario


if __name__ == "__main__":
    setup_logging()
    main()
