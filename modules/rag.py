# rag.py

import logging
from importlib import import_module
from operator import itemgetter
from typing import Any, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from modules.utils import load_configuration

# Load configuration globally to be reused in various functions
config = load_configuration()


def setup_rag_chain(
    file_urls: List[str], llm: ChatOpenAI, vector_index_path: str
) -> Any:
    """Set up a Retrieval-Augmented Generation (RAG) chain using specified files and save the resulting vector index."""
    logging.info(f"Processing file URLs: {file_urls}")
    docs = []

    # Iterate over each file URL and process documents
    for file_url in file_urls:
        file_type = file_url.split(".")[-1]
        logging.info(f"Setting up RAG chain for {file_type} document at {file_url}")

        # Dynamically load document loader based on file type
        loader_class = config["document_loaders"].get(file_type)
        if loader_class:
            loader_module = import_module("langchain_community.document_loaders")
            DocumentLoader = getattr(loader_module, loader_class)
            loaded_docs = DocumentLoader(file_url).load()
            docs.extend(loaded_docs)
            logging.info(f"Documents loaded from {file_url}: {len(loaded_docs)}")
        else:
            error_msg = f"Unsupported file type or loader not defined for {file_url}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    # Process all loaded documents for vector store creation
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        splits = text_splitter.split_documents(docs)

        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(splits, embedding_model)
        vectorstore.save_local(vector_index_path)

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


def load_vectorstore(vector_index_path: str, llm: ChatOpenAI) -> Any:
    """Load the vector store from a file and set up a RAG chain for retrieval and answering."""
    logging.info(f"Loading vector store from: {vector_index_path}")
    embedding_model = OpenAIEmbeddings(model=config["embedding_model"])
    vectorstore = FAISS.load_local(
        vector_index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    # Setup RAG chain with the loaded vector store
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
    logging.info("Vector store loaded and RAG chain setup complete")
    return rag_chain
