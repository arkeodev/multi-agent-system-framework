# model_service.py

import os

import streamlit as st
from pydantic import ValidationError

from config.config import ModelConfig


def instantiate_llm(config: ModelConfig, api_key: str):
    """Instantiate the language learning model based on the provided configuration and API key."""
    try:
        params = {"model": config.model_name, "temperature": config.temperature}
        if config.model_company == "openai":
            params["openai_api_key"] = api_key
            return config.chat_model_class(**params)
        elif config.model_company == "ollama":
            return config.chat_model_class(**params)
        else:
            st.error("Selected model configuration is not supported.")
            return None
    except ValidationError as e:
        st.error("Configuration Error: Check your model parameters and types.")
        return None


def ensure_api_key_is_set(model_type: str) -> str:
    """Ensure that the appropriate API key is set based on the selected model type and return it if available."""
    api_key_env_vars = {
        "openai": "OPENAI_API_KEY",
    }
    env_var_name = api_key_env_vars.get(model_type)
    if not env_var_name:
        return None
    api_key = os.getenv(env_var_name)
    if not api_key:
        api_key = st.text_input(
            f"Enter your API Key for {model_type.capitalize()}:", type="password"
        )
        if api_key:
            os.environ[env_var_name] = api_key
    return api_key
