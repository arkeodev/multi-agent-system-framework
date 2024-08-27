# rag.py

import logging
from operator import itemgetter
from typing import Any, List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from services.document_service import load_documents
from services.url_service import WebScraper


def setup_rag_chain(files_path_list: List[str], url: str, llm: ChatOpenAI) -> Any:
    """
    Set up a Retrieval-Augmented Generation (RAG) chain using either loaded
    documents or documents extrcted from url and save the resulting vector index.
    """
    logging.info(f"Setting up RAG chain for: {files_path_list or url}")

    # Load documents using the new document_loader module
    docs = get_documents(files_path_list, url)

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        splits = text_splitter.split_documents(docs)

        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(splits, embedding_model)
        retriever = vectorstore.as_retriever()
        rag_prompt = ChatPromptTemplate.from_template(
            "Context: {context}\n\nQuery: {question}\n\nUse the context to answer the query. If you can't answer, say you don't know."
        )
        rag_chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        logging.info("RAG chain setup complete")
        return rag_chain
    else:
        raise ValueError("No documents were loaded, RAG chain setup cannot proceed.")


def get_documents(
    files_uploaded: Optional[List[str]] = None, url: Optional[str] = None
) -> List[Document]:
    """Load documents from either files uploads or a URL."""
    documents = []

    if files_uploaded:
        logging.info(f"Loading documents from file uploads: {files_uploaded}")
        documents.extend(load_documents(files_uploaded))

    if url:
        logging.info(f"Loading documents from URL: {url}")
        web_scraper = WebScraper()
        documents.extend(web_scraper.scrape_website(url))

    return documents
