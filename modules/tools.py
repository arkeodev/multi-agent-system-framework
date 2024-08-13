# tools.py

import logging
from typing import Any, Type

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool


class RagInput(BaseModel):
    """Defines the input schema for queries to the RAG tool."""

    query: str = Field(description="Query to search in the RAG documents")


class RagTool(BaseTool):
    """A tool for retrieving data from the documents using a Retrieval-Augmented Generation (RAG) chain."""

    name = "RagTool"
    description = "Fetches documents data using a RAG chain."
    args_schema: Type[BaseModel] = RagInput
    rag_chain: Any

    def __init__(self, rag_chain: Any, **kwargs):
        super().__init__(**kwargs)
        self.rag_chain = rag_chain
        logging.info("RagTool initialized with RAG chain.")

    def _run(self, query: str) -> str:
        """Synchronously fetch data from the RAG documents using the provided query."""
        try:
            result = self.rag_chain.invoke({"question": query})
            logging.debug(f"Query result: {result}")
            return result
        except Exception as e:
            logging.error(f"Failed to retrieve data for query '{query}': {str(e)}")
            raise RuntimeError(f"Error processing query: {str(e)}") from e

    async def _arun(self, query: str) -> str:
        """Asynchronously fetch data from the documents, mirroring the synchronous _run method for future use."""
        # Placeholder for async behavior, to be implemented when necessary
        logging.warning("Async run not implemented yet.")
        raise NotImplementedError("Async operations are not supported yet.")
