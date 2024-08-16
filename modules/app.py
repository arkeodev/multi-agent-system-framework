# app.py

import logging
import os
from typing import Any, List, Optional

from modules.agent import AgentModel, create_agents
from modules.config.config import FileUploadConfig, URLConfig
from modules.execution import execute_scenario
from modules.rag import load_vectorstore, setup_rag_chain
from modules.supervisor import create_team_supervisor
from modules.tools import RagTool


class App:
    def __init__(
        self,
        llm: Any,
        recursion_limit: int,
        agent_config: dict,
        file_config: Optional[FileUploadConfig],
        url_config: Optional[URLConfig],
    ):
        logging.info("Initializing the App with custom configuration")
        self.llm = llm
        self.recursion_limit = recursion_limit
        self.agent_config = agent_config
        self.vector_index_path = "vector_index.faiss"
        self.file_config = file_config
        self.url_config = url_config

    def setup_and_run_scenario(self, recursion_limit: int, message_placeholder) -> None:
        """Set up agents, create the supervisor, and run the given scenario while updating the Streamlit UI."""
        logging.info("Setting up and running scenario")
        agents = self.setup_agents()
        agent_dict = {agent.role_name: agent.agent for agent in agents}
        supervisor_agent = self.create_supervisor()
        messages = []

        for message in execute_scenario(
            self.agent_config, agent_dict, supervisor_agent, recursion_limit
        ):
            messages.append(message)
            message_placeholder.write(
                "\n".join(messages)
            )  # Update the placeholder with current messages

    def setup_agents(self) -> List[AgentModel]:
        """Set up the agents using the RAG tool chain."""
        logging.info("Setting up agents")
        if os.path.exists(self.vector_index_path):
            rag_chain = load_vectorstore(self.vector_index_path, self.llm)
        else:
            rag_chain = setup_rag_chain(
                self.file_config.files,
                self.llm,
                self.vector_index_path,
            )
        rag_tool = RagTool(rag_chain=rag_chain)
        agents: List[AgentModel] = create_agents(
            self.llm, [rag_tool], self.agent_config
        )
        return agents

    def create_supervisor(self) -> Any:
        """Create and return a supervisor agent configured with system prompts and members.`"""
        team_supervisor = create_team_supervisor(self.llm, self.agent_config)
        logging.info("Supervisor agent is created")
        return team_supervisor
