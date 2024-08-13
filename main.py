# main.py

import os

import streamlit as st
from langchain_openai import ChatOpenAI

from modules.app import App
from modules.utils import (
    load_agent_config,
    load_configuration,
    set_api_keys,
    setup_logging,
)


def main():
    """Main function to initialize and run the Streamlit UI."""
    st.set_page_config(layout="wide")
    st.title("Document Scraper - Multi-Agent System")

    # UI Setup
    left_column, _, right_column = st.columns([1, 0.1, 2.6])

    with left_column:
        model_choice, api_key, recursion_limit = display_ui()

    with right_column:
        scenario = display_scenario()

    llm = ChatOpenAI(
        model=model_choice, api_key=api_key
    )  # Ensure API key is correctly used
    app = App(llm, recursion_limit)

    # Trigger scenario execution
    if st.button("Run Scenario"):
        messages = app.setup_and_run_scenario(scenario, recursion_limit)
        st.write("\n".join(messages))


def display_ui() -> tuple:
    """Displays the user interface for model selection and API key input."""
    config = load_configuration()

    if not os.path.exists(".env"):
        api_key = st.text_input("Enter your OpenAI API Key:")
    else:
        set_api_keys()
        api_key = os.getenv("OPENAI_API_KEY")

    model_choice = st.selectbox("Select the model to use:", config["models"], index=0)
    recursion_limit = st.number_input(
        "Set Recursion Limit:", min_value=10, max_value=100, value=25
    )

    return model_choice, api_key, recursion_limit


def display_scenario() -> str:
    """Displays the scenario text input area."""
    agent_config = load_agent_config()
    scenario = st.text_area("Scenario Text", agent_config["scenario"], height=300)
    return scenario


if __name__ == "__main__":
    setup_logging()
    main()
