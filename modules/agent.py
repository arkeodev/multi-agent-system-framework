# agent.py
import logging
from typing import List

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.pydantic_v1 import BaseModel, validator
from langchain.tools.base import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from modules.utils import load_agent_config


class AgentModel(BaseModel):
    role_name: str
    agent: Runnable

    @validator("role_name", allow_reuse=True)
    def check_role_name(cls, v):
        if not v:
            raise ValueError("Role name must not be empty")
        return v

    class Config:
        arbitrary_types_allowed = True


def create_agents(llm: ChatOpenAI, tools: List[BaseTool]) -> List[AgentModel]:
    logging.info("Creating agents...")
    try:
        agent_config = load_agent_config()
    except Exception as e:
        logging.error(f"Failed to load agent configuration: {e}")
        raise
    agents = []

    for role in agent_config["roles"]:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", role["prompt"]),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        logging.debug(f"Prompt setup: {prompt}")

        agent = create_openai_functions_agent(llm, tools, prompt)
        logging.debug(f"Agent executor setup: {agent}")

        agent = AgentModel(
            role_name=role["name"],
            agent=AgentExecutor(agent=agent, tools=tools),
        )
        agents.append(agent)
        logging.debug(f"Agent setup: {agent}")

    logging.info("Agents created successfully.")
    return agents
