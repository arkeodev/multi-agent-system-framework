# api_service.py

import requests
from langchain.schema import Document


def process_api_and_add_to_faiss(api_url: str, faiss_index):
    response = requests.get(api_url)
    if response.status_code == 200:
        content = response.text
        doc = Document(page_content=content, metadata={"source": api_url})
        faiss_index.add_texts([doc.page_content], metadatas=[doc.metadata])
    else:
        raise Exception(f"Failed to fetch API data: {response.status_code}")
