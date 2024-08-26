# file_utils.py

import logging
from pathlib import Path


def save_uploaded_file(uploaded_file, save_dir="/tmp"):
    """Saves an uploaded file to the specified directory and returns the path."""
    try:
        save_directory = Path(save_dir)
        save_directory.mkdir(parents=True, exist_ok=True)
        file_path = save_directory / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        return file_path
    except Exception as e:
        logging.error(f"Failed to save uploaded file: {str(e)}")
        raise
