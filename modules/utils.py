# utils.py
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse.callback import CallbackHandler


def set_api_keys():
    """Loads and sets necessary API keys for OpenAI and LangFuse."""
    logging.info("Loading API keys from .env file.")
    load_dotenv()

    if os.getenv("OPENAI_API_KEY"):
        logging.info("OpenAI API key loaded successfully.")

    if all(
        os.getenv(key)
        for key in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    ):
        logging.info("LangFuse API keys loaded successfully.")


def setup_langfuse_keys(pk: str, sk: str, host: str) -> CallbackHandler:
    """Creates a LangFuse CallbackHandler with the provided keys."""
    return CallbackHandler(public_key=pk, secret_key=sk, host=host)


def setup_logging():
    """Sets up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def read_json(filepath: str) -> dict:
    """Reads a JSON file and returns it as a dictionary."""
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}  # Return an empty dictionary as fallback
    except json.JSONDecodeError:
        return {}  # Return an empty dictionary as fallback


def format_json(data: dict) -> str:
    """Takes a dictionary and returns it as a prettily formatted JSON string."""
    return json.dumps(data, indent=4)  # Pretty print the JSON


def save_uploaded_file(uploaded_file, save_dir="/tmp"):
    """
    Saves an uploaded file to the specified directory and returns the path.
    Works across Unix, macOS, and Windows.
    """
    try:
        save_directory = Path(save_dir)
        save_directory.mkdir(parents=True, exist_ok=True)
        file_path = save_directory / uploaded_file.name

        # Write the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        return file_path
    except Exception as e:
        logging.error(f"Failed to save uploaded file: {str(e)}")
        raise
