# rag.py

import logging
from operator import itemgetter
from typing import Any, List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from modules.document_loader import load_documents


def setup_rag_chain(
    file_paths: List[str], llm: ChatOpenAI, vector_index_path: str
) -> Any:
    """Set up a Retrieval-Augmented Generation (RAG) chain using loaded documents and save the resulting vector index."""
    logging.info(f"Setting up RAG chain for files: {file_paths}")

    # Load documents using the new document_loader module
    docs = load_documents(file_paths)

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
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
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
