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
    """Model representing an agent with a specific role and its executable agent logic."""

    role_name: str
    agent: Runnable

    @validator("role_name", allow_reuse=True)
    def check_role_name(cls, v: str) -> str:
        """Validate that the role name is not empty."""
        if not v:
            raise ValueError("Role name must not be empty")
        return v

    class Config:
        arbitrary_types_allowed = True


def create_agents(llm: ChatOpenAI, tools: List[BaseTool]) -> List[AgentModel]:
    """Create agents for each role defined in the agent configuration."""
    logging.info("Creating agents...")
    try:
        agent_config = load_agent_config()
    except Exception as e:
        logging.error(f"Failed to load agent configuration: {e}")
        raise

    agents = []
    for role in agent_config["roles"]:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", role["prompt"]),
                    MessagesPlaceholder(variable_name="messages"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            logging.debug(f"Prompt setup for {role['name']}: {prompt}")

            agent = create_openai_functions_agent(llm, tools, prompt)
            agent_model = AgentModel(
                role_name=role["name"],
                agent=AgentExecutor(agent=agent, tools=tools),
            )
            agents.append(agent_model)
            logging.info(f"Agent setup for {role['name']} completed successfully.")
        except Exception as e:
            logging.error(f"Failed to create agent for role {role['name']}: {e}")
            continue

    return agents
