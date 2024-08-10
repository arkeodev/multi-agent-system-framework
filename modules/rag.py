# rag.py
import logging
from operator import itemgetter
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def setup_rag_chain(pdf_url: str, collection_name: str, llm: ChatOpenAI, vector_index_path: str) -> Any:
    """Set up the Retrieval-Augmented Generation (RAG) chain and save vector index."""
    # Load and process document
    logging.info(f"Loading document from: {pdf_url}")
    docs = PyMuPDFLoader(pdf_url).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)
    logging.info(f"Document split into {len(splits)} parts")

    # Create vector store
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(
        splits, embedding_model
    )
    vectorstore.save_local(vector_index_path)

    retriever = vectorstore.as_retriever()

    # Create RAG prompt
    rag_prompt = ChatPromptTemplate.from_template(
        "Context: {context}\n\nQuery: {question}\n\nUse the context to answer the query. If you can't answer, say you don't know."
    )

    # Create RAG chain
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
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(vector_index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    # Create RAG prompt
    rag_prompt = ChatPromptTemplate.from_template(
        "Context: {context}\n\nQuery: {question}\n\nUse the context to answer the query. If you can't answer, say you don't know."
    )

    # Create RAG chain
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
