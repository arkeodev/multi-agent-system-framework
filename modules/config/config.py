from typing import Annotated, List

from pydantic import BaseModel, HttpUrl, constr


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
    files: list  # This will be filled with file paths or identifiers


StrippedString = Annotated[str, constr(strip_whitespace=True, min_length=1)]


class URLConfig(BaseModel):
    url: HttpUrl
    exclusion_pattern: StrippedString
    max_depth: int
