"""
utils.py

This module handles the configuration and environment setup for the project.
"""

import logging
import os

from dotenv import load_dotenv


def set_api_keys():
    """Sets the necessary API keys for OpenAI and SerpAPI."""
    logging.info("Loading API keys from .env file.")
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        logging.error("API keys for OpenAI and SerpAPI must be set in the .env file.")
        raise ValueError(
            "API keys for OpenAI and SerpAPI must be set in the .env file."
        )

    os.environ["OPENAI_API_KEY"] = openai_api_key
    logging.info("API keys loaded successfully.")


def setup_logging():
    """Sets up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,  # Set the default logging level to INFO.
        format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format.
        handlers=[logging.StreamHandler()],
    )
