# scenario_generator.py

import json
import logging
from typing import List, Optional

from langchain.schema import Document
from langchain_openai import ChatOpenAI

from modules.config.config import FileUploadConfig, URLConfig
from modules.document_loader import load_documents
from modules.url_handler import scrape_website
from modules.utils import format_json


def generate_scenario_config(llm: ChatOpenAI, documents: List[Document]) -> str:
    """Generate a scenario configuration file using the LLM based on the provided documents."""
    logging.info("Generating scenario configuration from documents")

    # Combine all document contents
    combined_content = "\n\n".join([doc.page_content for doc in documents])

    # Prompt LLM to generate agent configuration based on combined content
    prompt = (
        "You are tasked with generating a JSON configuration for a multi-agent scenario. "
        "The configuration should define the roles, prompts, and scenario details for the agents involved. "
        "Use the example below as a template for structure only. Do not copy any of the values, only use the structure:\n\n"
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
        "}\n\n"
        f"Based on the following content, generate a similar JSON configuration:\n{combined_content}"
    )

    response = llm(prompt)
    logging.info(f"Generated scenario configuration: {response}")

    raw_json_string = response.content.strip()
    cleaned_json_string = raw_json_string.strip("```json").strip("```")
    try:
        json_data = json.loads(cleaned_json_string)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON: {e}")
        return "{}"  # Return an empty JSON object as fallback
    formatted_json = format_json(json_data)
    return formatted_json


def load_documents_for_scenario(
    file_upload_config: Optional[FileUploadConfig], url_config: Optional[URLConfig]
) -> List[Document]:
    """Load documents from either file uploads or a URL for scenario generation."""
    documents = []

    if file_upload_config:
        documents.extend(load_documents(file_upload_config.files))

    if url_config:
        documents.extend(
            scrape_website(
                url_config.url, [url_config.exclusion_pattern], url_config.max_depth
            )
        )

    return documents
