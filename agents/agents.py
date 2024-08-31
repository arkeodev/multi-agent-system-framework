# agents.py

import logging
from enum import Enum, auto
from typing import List

from langchain.agents import (
    AgentExecutor,
    Tool,
    create_openai_functions_agent,
    create_react_agent,
)
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, validator
from langchain.tools.base import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun, HumanInputRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.runnables import Runnable
from langchain_experimental.tools.python.tool import PythonREPLTool


class StandardAgent:
    """
    StandardAgent represents a base agent class with predefined tools and memory.

    This class encapsulates the common functionality for creating and managing
    standard agents. It initializes with a language model, a list of tools,
    and a name. The agent uses a conversation buffer memory to maintain context
    across interactions.
    """

    def __init__(self, llm, tools: List[Tool], name: str):
        self.llm = llm
        self.tools = tools
        self.name = name
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        logging.info(f"Initialized StandardAgent: {name}")

    def get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"You are {self.name}, an AI assistant. Use the following tools to answer the human's questions: {{tools}}. The available tools are: {{tool_names}}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{{input}}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def get_agent(self):
        prompt = self.get_prompt()
        tool_names = [tool.name for tool in self.tools]
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
            input_variables=["input", "chat_history", "agent_scratchpad"],
            tool_names=tool_names,
        )


class StandardAgentRegistry:
    class StandardAgents(Enum):
        SEARCH = auto()
        WIKIPEDIA = auto()
        PYTHON_REPL = auto()
        HUMAN_INTERACTION = auto()

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StandardAgentRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.agent_names = {
            self.StandardAgents.SEARCH: "Search Agent",
            self.StandardAgents.WIKIPEDIA: "Wikipedia Agent",
            self.StandardAgents.PYTHON_REPL: "Python Agent",
            self.StandardAgents.HUMAN_INTERACTION: "Human Interaction Agent",
        }

    def get_name(self, agent_type):
        return self.agent_names[agent_type]

    def get_all_names(self):
        return list(self.agent_names.values())


agent_registry = StandardAgentRegistry()


def create_tool_based_agents(llm):
    """Create standard agents with predefined tools."""
    logging.info("Creating tool-based agents")

    tools = {
        StandardAgentRegistry.StandardAgents.SEARCH: Tool(
            name=agent_registry.get_name(StandardAgentRegistry.StandardAgents.SEARCH),
            func=DuckDuckGoSearchRun().run,
            description="useful for when you need to answer questions about current events",
        ),
        StandardAgentRegistry.StandardAgents.WIKIPEDIA: Tool(
            name=agent_registry.get_name(
                StandardAgentRegistry.StandardAgents.WIKIPEDIA
            ),
            func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
            description="useful for when you need to query general knowledge",
        ),
        StandardAgentRegistry.StandardAgents.PYTHON_REPL: Tool(
            name=agent_registry.get_name(
                StandardAgentRegistry.StandardAgents.PYTHON_REPL
            ),
            func=PythonREPLTool().run,
            description="useful for when you need to run Python code to solve a problem",
        ),
        StandardAgentRegistry.StandardAgents.HUMAN_INTERACTION: Tool(
            name=agent_registry.get_name(
                StandardAgentRegistry.StandardAgents.HUMAN_INTERACTION
            ),
            func=HumanInputRun().run,
            description="useful when you need to ask a human for additional information",
        ),
    }

    agents = [StandardAgent(llm, [tool], name) for name, tool in tools.items()]

    logging.info(f"Created {len(agents)} tool-based agents")
    return agents


class RoleBasedAgentModel(BaseModel):
    """Represents an agent with a specific role and its executable logic."""

    role_name: str
    agent: Runnable

    @validator("role_name", allow_reuse=True)
    def check_role_name(cls, v: str) -> str:
        """Ensure the role name is not empty."""
        if not v:
            raise ValueError("Role name must not be empty")
        return v

    class Config:
        arbitrary_types_allowed = True


class RoleBasedAgentFactory:
    """
    Factory class for creating role based agent instances.

    This class provides static methods to create different types of agents
    based on the provided language model, tools, and role configurations.
    It encapsulates the agent creation logic, allowing for centralized
    management and easy extension of agent types.
    """

    @staticmethod
    def create_agent(llm, tools: List[BaseTool], role: dict) -> RoleBasedAgentModel:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", role.get("prompt")),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_openai_functions_agent(llm, tools, prompt)
        return RoleBasedAgentModel(
            role_name=role["name"], agent=AgentExecutor(agent=agent, tools=tools)
        )


def create_role_based_agents(
    llm, tools: List[BaseTool], roles: List[dict]
) -> List[RoleBasedAgentModel]:
    """Create role-based agents with custom roles and tools."""
    logging.info("Creating role-based agents")
    agents = [RoleBasedAgentFactory.create_agent(llm, tools, role) for role in roles]
    logging.info(f"Created {len(agents)} role-based agents")
    return agents
