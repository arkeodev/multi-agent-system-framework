# tools.py

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool


class ToolInput(BaseModel):
    """Defines the input schema for queries to the tool."""

    query: str = Field(description="Query to be processed by the tool")


class AbstractTool(BaseTool, ABC):
    """Base class for all tools, providing a common interface."""

    name: str
    description: str
    args_schema: Any = ToolInput

    @abstractmethod
    def _run(self, query: str) -> str:
        """Synchronously fetch data from the tool using the provided query."""
        pass

    async def _arun(self, query: str) -> str:
        """Asynchronously fetch data from the tool, mirroring the synchronous _run method for future use."""
        # Default to _run if not implemented
        logging.warning("Async run not implemented yet.")
        return self._run(query)


class RagTool(AbstractTool):
    """A tool for retrieving data from the documents using a Retrieval-Augmented Generation (RAG) chain."""

    name = "RagTool"
    description = "Fetches documents data using a RAG chain."
    rag_chain: Any

    def __init__(self, rag_chain: Any, **kwargs):
        super().__init__(**kwargs)
        self.rag_chain = rag_chain

    def _run(self, query: str) -> str:
        """Synchronously fetch data from the RAG documents using the provided query."""
        try:
            result = self.rag_chain.invoke({"question": query})
            logging.debug(f"Query result: {result}")
            return result
        except Exception as e:
            raise RuntimeError(f"Error processing query: {str(e)}") from e
