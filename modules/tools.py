# tools.py
from typing import Any, Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool


class HandbookInput(BaseModel):
    query: str = Field(description="Query to search in the handbook")


class HandbookTool(BaseTool):
    name = "HandbookTool"
    description = "Fetches handbook data using a RAG chain."
    args_schema: Type[BaseModel] = HandbookInput
    handbook_rag_chain: Any

    def __init__(self, handbook_rag_chain: Any, **kwargs):
        super().__init__(**kwargs)
        self.handbook_rag_chain = handbook_rag_chain

    def _run(self, query: str) -> str:
        """Synchronously fetch data from the handbook using the provided query."""
        return self.handbook_rag_chain.invoke({"question": query})

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """This method is a placeholder for asynchronous behavior, currently not implemented."""
        raise NotImplementedError(f"{self.name} does not support async operations yet.")
