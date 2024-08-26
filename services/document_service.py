# document_service.py

import logging
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader, UnstructuredEPubLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.xml import UnstructuredXMLLoader

# Mapping of file types to corresponding document loaders
DEFAULT_LOADERS = {
    "pdf": PyMuPDFLoader,
    "csv": CSVLoader,
    "epub": UnstructuredEPubLoader,
    "md": UnstructuredMarkdownLoader,
    "json": JSONLoader,
    "xml": UnstructuredXMLLoader,
    "txt": TextLoader,
}


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load documents based on their file types and return them in LangChain's Document format with metadata."""
    docs = []

    for file_path in file_paths:
        logging.info(f"Loading file: {file_path}")
        try:
            # Determine the file type based on the file extension
            file_type = file_path.split(".")[-1].lower()
            logging.info(f"Loading {file_type} document from {file_path}")

            # Get the appropriate loader class for the file type
            loader_class = DEFAULT_LOADERS.get(file_type)
            if loader_class:
                # Instantiate the loader and load the documents
                loader = loader_class(file_path)
                loaded_docs = loader.load()
            else:
                logging.error(f"No loader found for file type: {file_type}")
                raise ValueError(f"Unsupported file type: {file_type}")

            # Add metadata to each loaded document
            for doc in loaded_docs:
                doc.metadata = generate_metadata(doc, file_path)
            docs.extend(loaded_docs)

        except ValueError as ve:
            logging.error(f"ValueError: {str(ve)}")
        except Exception as e:
            logging.error(f"Failed to load document from {file_path}: {str(e)}")
            continue  # Skip the current file and proceed with the next one

    return docs


def generate_metadata(document: Document, file_path: str) -> dict:
    """Generate metadata for the document based on the file path and document content."""
    metadata = {
        "source": file_path,
        "content_length": len(document.page_content),
    }
    return metadata
