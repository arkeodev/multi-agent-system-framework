# config.py

from typing import Any, Dict, List, Optional

from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, HttpUrl

# Constants
AGENT_SUPERVISOR = "supervisor"
VECTOR_INDEX_PATH = "vector_index.faiss"
ROUTE_NAME = "route"
FINISH = "FINISH"


class Role(BaseModel):
    name: str
    prompt: str


class SupervisorPrompts(BaseModel):
    initial: str
    decision: str


class AgentConfig(BaseModel):
    supervisor_prompts: SupervisorPrompts
    members: List[str]
    roles: List[Role]
    scenario: str


class FileUploadConfig(BaseModel):
    files: List[str]  # This will be filled with file paths or identifiers


class URLConfig(BaseModel):
    url: HttpUrl


class ModelConfig(BaseModel):
    model_company: str
    model_name: str
    temperature: float
    chat_model_class: Any
    api_key: Optional[str] = None

    class Config:
        protected_namespaces = ()


# Configuration for models with direct class references
model_config_dict: Dict = {
    "openai": {
        "gpt-4o-mini": ModelConfig(
            model_company="openai",
            model_name="gpt-4o-mini",
            temperature=0.3,
            chat_model_class=ChatOpenAI,
        ),
        "gpt-4o": ModelConfig(
            model_company="openai",
            model_name="gpt-4o",
            temperature=0.3,
            chat_model_class=ChatOpenAI,
        ),
    },
    "ollama": {
        "llama3.1:8b": ModelConfig(
            model_company="ollama",
            model_name="llama3.1:8b",
            temperature=0.3,
            chat_model_class=ChatOllama,
        ),
        "llama3.1:70b": ModelConfig(
            model_company="ollama",
            model_name="llama3.1:70b",
            temperature=0.3,
            chat_model_class=ChatOllama,
        ),
    },
}
