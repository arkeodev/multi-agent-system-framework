from typing import List

from pydantic import BaseModel, HttpUrl


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
    scenario_source_documents: List[HttpUrl]
    scenario: str
