# json_utils.py

import json
import logging


def read_json(filepath: str) -> dict:
    """Reads a JSON file and returns it as a dictionary."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return {}  # Return an empty dictionary as fallback
    except json.JSONDecodeError:
        logging.error(f"JSON decode error in file: {filepath}")
        return {}  # Return an empty dictionary as fallback


def format_json(data: dict, ensure_ascii=False) -> str:
    """Takes a dictionary and returns it as a prettily formatted JSON string."""
    return json.dumps(
        data, indent=4, ensure_ascii=ensure_ascii
    )  # Pretty print the JSON
