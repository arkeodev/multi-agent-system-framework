# rag.py

import logging
from operator import itemgetter
from typing import Any, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from services.document_service import load_documents


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
