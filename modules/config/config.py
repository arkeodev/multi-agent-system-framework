# config.py

from typing import Annotated, Any, Dict, List, Optional

from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, HttpUrl, constr

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


StrippedString = Annotated[str, constr(strip_whitespace=True, min_length=1)]


class URLConfig(BaseModel):
    url: HttpUrl
    exclusion_pattern: StrippedString
    max_depth: int


class ModelConfig(BaseModel):
    model_type: str
    model_name: str
    temperature: float
    chat_model_class: Any
    api_key: Optional[str] = None


# Configuration for models with direct class references
model_config_dict: Dict[str, Dict[str, ModelConfig]] = {
    "openai": {
        "gpt-4o-mini": ModelConfig(
            model_type="openai",
            model_name="gpt-4o-mini",
            temperature=0.3,
            chat_model_class=ChatOpenAI,
        ),
        "gpt-4o": ModelConfig(
            model_type="openai",
            model_name="gpt-4o",
            temperature=0.3,
            chat_model_class=ChatOpenAI,
        ),
    },
    "ollama": {
        "llama3.1:8b": ModelConfig(
            model_type="ollama",
            model_name="llama3.1:8b",
            temperature=0.3,
            chat_model_class=ChatOllama,
        ),
        "llama3.1:70b": ModelConfig(
            model_type="ollama",
            model_name="llama3.1:70b",
            temperature=0.3,
            chat_model_class=ChatOllama,
        ),
    },
}
