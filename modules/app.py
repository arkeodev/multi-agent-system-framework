# app.py

import logging
from typing import Any, List, Optional

from langfuse.callback import CallbackHandler

from modules.agent import AgentModel, create_agents
from modules.config.config import VECTOR_INDEX_PATH, FileUploadConfig, URLConfig
from modules.execution import execute_scenario
from modules.rag import setup_rag_chain
from modules.supervisor import create_team_supervisor
from modules.tools import RagTool


class App:
    def __init__(
        self,
        llm: Any,
        recursion_limit: int,
        agent_config: dict,
        file_config: Optional[FileUploadConfig] = None,
        url_config: Optional[URLConfig] = None,
    ):
        """Initializes the application with LLM, configuration and limits."""
        self.llm = llm
        self.recursion_limit = recursion_limit
        self.agent_config = agent_config
        self.file_config = file_config
        self.url_config = url_config

    def setup_and_run_scenario(
        self, message_placeholder, langfuse_handler: Optional[CallbackHandler]
    ) -> List[str]:
        """Sets up agents and runs the scenario, processing messages interactively."""
        logging.info("Setting up and running scenario")
        agents = self.setup_agents()
        agent_dict = {agent.role_name: agent.agent for agent in agents}
        supervisor_agent = self.create_supervisor()
        messages = []

        last_displayed_message = ""
        try:
            for message in execute_scenario(
                self.agent_config["scenario"],
                agent_dict,
                supervisor_agent,
                self.recursion_limit,
                langfuse_handler,
            ):
                logging.debug(f"Received message: {message}")
                if message != last_displayed_message:
                    if messages:
                        messages.append("\n")
                    messages.append(message)
                    # Update the display with each new message and spaces
                    message_placeholder.write("\n".join(messages))
                    last_displayed_message = message
        except Exception as e:
            logging.error(f"Error during scenario execution: {e}")
            raise RuntimeError(f"Failed to execute scenario due to: {e}")

        return messages

    def setup_agents(self) -> List[AgentModel]:
        """Configures agents based on the provided configuration."""
        logging.info("Setting up agents")
        rag_chain = setup_rag_chain(
            self.file_config.files,
            self.llm,
            VECTOR_INDEX_PATH,
        )
        rag_tool = RagTool(rag_chain=rag_chain)
        agents: List[AgentModel] = create_agents(
            self.llm, [rag_tool], self.agent_config["roles"]
        )
        return agents

    def create_supervisor(self) -> Any:
        """Creates a supervisor agent configured with specific system prompts and member roles."""
        team_supervisor = create_team_supervisor(
            self.llm,
            self.agent_config["members"],
            self.agent_config["supervisor_prompts"],
        )
        logging.info("Supervisor agent created")
        return team_supervisor
