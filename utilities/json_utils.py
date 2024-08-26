# json_utils.py

import json
import logging


def read_json(filepath: str) -> dict:
    """Reads a JSON file and returns it as a dictionary."""
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return {}  # Return an empty dictionary as fallback
    except json.JSONDecodeError:
        logging.error(f"JSON decode error in file: {filepath}")
        return {}  # Return an empty dictionary as fallback


def format_json(data: dict) -> str:
    """Takes a dictionary and returns it as a prettily formatted JSON string."""
    return json.dumps(data, indent=4)  # Pretty print the JSON
