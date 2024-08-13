# tools.py

import logging
from typing import Any, Type

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool


class HandbookInput(BaseModel):
    """Defines the input schema for queries to the handbook tool."""

    query: str = Field(description="Query to search in the handbook")


class HandbookTool(BaseTool):
    """A tool for retrieving data from a handbook using a Retrieval-Augmented Generation (RAG) chain."""

    name = "HandbookTool"
    description = "Fetches handbook data using a RAG chain."
    args_schema: Type[BaseModel] = HandbookInput
    handbook_rag_chain: Any

    def __init__(self, handbook_rag_chain: Any, **kwargs):
        super().__init__(**kwargs)
        self.handbook_rag_chain = handbook_rag_chain
        logging.info("HandbookTool initialized with RAG chain.")

    def _run(self, query: str) -> str:
        """Synchronously fetch data from the handbook using the provided query."""
        try:
            result = self.handbook_rag_chain.invoke({"question": query})
            logging.debug(f"Query result: {result}")
            return result
        except Exception as e:
            logging.error(f"Failed to retrieve data for query '{query}': {str(e)}")
            raise RuntimeError(f"Error processing query: {str(e)}") from e

    async def _arun(self, query: str) -> str:
        """Asynchronously fetch data from the handbook, mirroring the synchronous _run method for future use."""
        # Placeholder for async behavior, to be implemented when necessary
        logging.warning("Async run not implemented yet.")
        raise NotImplementedError("Async operations are not supported yet.")
