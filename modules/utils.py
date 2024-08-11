# utils.py
import json
import logging
import os

from dotenv import load_dotenv


def load_configuration():
    """Load the configuration for models and document loaders."""
    with open("modules/config/app_config.json", "r") as config_file:
        config = json.load(config_file)
    return config


def load_agent_config():
    """Load the configuration for models and document loaders."""
    with open("modules/config/agent_config.json", "r") as config_file:
        config = json.load(config_file)
    return config


def set_api_keys():
    """Sets the necessary API keys for OpenAI and other APIs."""
    logging.info("Loading API keys from .env file.")
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.error("API key for OpenAI must be set in the .env file.")
        raise ValueError("API key for OpenAI must be set in the .env file.")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    logging.info("API keys loaded successfully.")


def setup_logging():
    """Sets up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
