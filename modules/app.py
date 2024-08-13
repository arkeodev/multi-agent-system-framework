# app.py

import logging
import os
from typing import Any, List

from modules.agent import AgentModel, create_agents
from modules.execution import execute_scenario
from modules.rag import load_vectorstore, setup_rag_chain
from modules.supervisor import create_team_supervisor
from modules.tools import RagTool
from modules.utils import load_agent_config


class App:
    def __init__(self, llm: Any, recursion_limit: int):
        logging.info("Initializing the App")
        self.llm = llm
        self.recursion_limit = recursion_limit
        self.agent_config = load_agent_config()
        self.vector_index_path = "vector_index.faiss"

    def setup_and_run_scenario(self, scenario: str, recursion_limit: int) -> List[str]:
        """Set up agents, create the supervisor, and run the given scenario."""
        logging.info("Setting up and running scenario")
        agents = self.setup_agents()
        agent_dict = {agent.role_name: agent.agent for agent in agents}
        supervisor_agent = self.create_supervisor()
        messages = execute_scenario(
            scenario, agent_dict, supervisor_agent, recursion_limit
        )
        return messages

    def setup_agents(self) -> List[AgentModel]:
        """Set up the agents using the RAG tool chain."""
        logging.info("Setting up agents")
        if os.path.exists(self.vector_index_path):
            rag_chain = load_vectorstore(self.vector_index_path, self.llm)
        else:
            rag_chain = setup_rag_chain(
                self.agent_config["document_urls"],
                self.llm,
                self.vector_index_path,
            )
        rag_tool = RagTool(rag_chain=rag_chain)
        agents: List[AgentModel] = create_agents(self.llm, [rag_tool])
        return agents

    def create_supervisor(self) -> Any:
        """Create and return a supervisor agent configured with system prompts and members.`"""
        logging.info("Creating supervisor agent")
        return create_team_supervisor(
            self.llm,
            self.agent_config["supervisor_prompts"]["initial"],
            self.agent_config["members"],
        )
