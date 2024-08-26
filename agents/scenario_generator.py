# scenario_generator.py

import json
import logging
from typing import List, Optional

from langchain.schema import Document
from langchain_openai import ChatOpenAI

from config.config import FileUploadConfig
from services.document_service import load_documents
from services.url_service import WebScraper
from utilities.json_utils import format_json


def clean_json_string(raw_string: str) -> str:
    """Cleans up the raw string returned by the LLM by removing markdown formatting."""
    if "```json" in raw_string:
        # Remove ```json and ``` or any surrounding whitespace
        cleaned_string = raw_string.split("```json")[-1].split("```")[0].strip()
    else:
        cleaned_string = raw_string.strip()
    return cleaned_string


def generate_scenario_config(llm: ChatOpenAI, documents: List[Document]) -> str:
    """Generate a scenario configuration file using the LLM based on the provided documents."""
    logging.info("Generating scenario configuration from documents")

    # Combine all document contents
    combined_content = "\n\n".join([doc.page_content for doc in documents])

    # Prompt LLM to generate agent configuration based on combined content
    messages = [
        (
            "system",
            "You are tasked with generating a JSON configuration for a multi-agent scenario. The configuration should define the roles, prompts, and scenario details for the agents involved. Use the example below as a template for structure only. Do not copy any of the values, only use the structure:",
        ),
        (
            "system",
            "Example Configuration Structure:\n"
            "{\n"
            '    "supervisor_prompts": {\n'
            '        "initial": "This is where the prompt for the initial instruction goes.",\n'
            '        "decision": "This is where the prompt for decision making goes."\n'
            "    },\n"
            '    "members": [\n'
            '        "Role 1",\n'
            '        "Role 2",\n'
            '        "Role 3"\n'
            "    ],\n"
            '    "roles": [\n'
            "        {\n"
            '            "name": "Role 1",\n'
            '            "prompt": "This is the specific prompt for Role 1."\n'
            "        },\n"
            "        {\n"
            '            "name": "Role 2",\n'
            '            "prompt": "This is the specific prompt for Role 2."\n'
            "        },\n"
            "        {\n"
            '            "name": "Role 3",\n'
            '            "prompt": "This is the specific prompt for Role 3."\n'
            "        }\n"
            "    ],\n"
            '    "scenario": "This is where the scenario description goes."\n'
            "}\n",
        ),
        (
            "human",
            f"Based on the following content, generate a similar JSON configuration:\n{combined_content}",
        ),
    ]

    try:
        response = llm.invoke(messages)
        logging.debug(f"Generated scenario configuration: {response}")
    except Exception as e:
        logging.error(f"Failed to generate scenario: {e}")
        return f"{e}"

    # Clean and parse the JSON string
    cleaned_json_string = clean_json_string(response.content)
    logging.debug(f"Cleaned json string is: {cleaned_json_string}")
    try:
        json_data = json.loads(cleaned_json_string)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON: {e}\nRaw string: {cleaned_json_string}")
        return "{The model can't generate a proper formatted scenario. Please try different model.}"

    # Format the JSON to be pretty-printed
    formatted_json = format_json(json_data)
    logging.debug(f"Formatted JSON configuration: {formatted_json}")
    logging.info(f"Scenario configuration generated.")
    return formatted_json


def load_documents_for_scenario(
    file_upload_config: Optional[FileUploadConfig], url: str
) -> List[Document]:
    """Load documents from either file uploads or a URL for scenario generation."""
    documents = []

    if file_upload_config:
        documents.extend(load_documents(file_upload_config.files))

    if url:
        web_scraper = WebScraper()
        documents.extend(web_scraper.scrape_website(url))

    return documents
