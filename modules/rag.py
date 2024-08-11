import logging

# Importing necessary modules dynamically
from importlib import import_module
from operator import itemgetter
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from modules.utils import load_configuration

config = load_configuration()


def setup_rag_chain(file_url: str, llm: ChatOpenAI, vector_index_path: str) -> Any:
    """Set up the Retrieval-Augmented Generation (RAG) chain and save vector index."""
    logging.info(f"File url is: {file_url}")
    file_type = file_url.split(".")[-1]  # Extract file extension
    logging.info(f"Setting up RAG chain for {file_type} document")
    loader_class = config["document_loaders"].get(file_type)

    # Dynamically import the correct loader based on file extension
    if loader_class:
        loader_module = import_module("langchain_community.document_loaders")
        DocumentLoader = getattr(loader_module, loader_class)
        docs = DocumentLoader(file_url).load()
        logging.info(
            f"Documents loaded: {len(docs)}"
        )  # Log the number of documents loaded
    else:
        logging.error("Unsupported file type or loader not defined")
        raise ValueError("Unsupported file type or loader not defined")

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


def load_vectorstore(vector_index_path: str, llm: ChatOpenAI) -> Any:
    """Load the vector store from a file."""
    logging.info(f"Loading vector store from: {vector_index_path}")
    embedding_model = OpenAIEmbeddings(model=config["embedding_model"])
    vectorstore = FAISS.load_local(
        vector_index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
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
