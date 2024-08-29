# app.py

import logging
from io import BytesIO
from typing import Any, List, Optional

from langchain_core.runnables.graph import MermaidDrawMethod
from langfuse.callback import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from PIL import Image

from agents.agents import RoleBasedAgentModel, create_role_based_agents
from agents.graph import create_graph
from agents.rag import setup_rag_chain
from agents.supervisor import create_team_supervisor
from agents.tools import RagTool
from config.config import FileUploadConfig
from core.execution import execute_graph


class App:
    def __init__(
        self,
        llm: Any,
        recursion_limit: int,
        agent_config: dict,
        file_config: Optional[FileUploadConfig] = None,
        url: Optional[str] = None,
        langfuse_handler: Optional[CallbackHandler] = None,
        agent_factory=create_role_based_agents,
        rag_tool_factory=RagTool,
        supervisor_factory=create_team_supervisor,
        graph_factory=create_graph,
    ):
        """Initializes the application with LLM, configuration and limits."""
        self.llm = llm
        self.recursion_limit = recursion_limit
        self.agent_config = agent_config
        self.file_config = file_config
        self.url = url
        self.langfuse_handler = langfuse_handler
        self.agent_factory = agent_factory
        self.rag_tool_factory = rag_tool_factory
        self.supervisor_factory = supervisor_factory
        self.graph_factory = graph_factory
        self._graph: CompiledStateGraph = None

    @property
    def graph(self):
        """Lazily creates and returns the graph if not already created."""
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    def build_graph(self):
        """Creates langgraph using the injected factories."""
        agents = self.setup_agents()
        agent_dict = {agent.role_name: agent.agent for agent in agents}
        supervisor_agent = self.create_supervisor()
        return self.graph_factory(agent_dict, supervisor_agent, self.llm).compile()

    def execute_graph(self, message_placeholder) -> List[str]:
        """Runs the graph, processing messages interactively."""
        logging.info("Setting up and running graph")
        messages = []
        last_displayed_message = ""
        try:
            for message in execute_graph(
                self.graph,
                self.agent_config["scenario"],
                self.recursion_limit,
                self.langfuse_handler,
            ):
                logging.debug(f"Received message: {message}")
                if message != last_displayed_message:
                    if messages:
                        messages.append("\n")
                    messages.append(message)
                    # Update the display with each new message and spaces
                    message_placeholder.write(
                        "\n".join(messages)
                    )  # Update the placeholder here
                    last_displayed_message = message
        except Exception as e:
            logging.error(f"Error during execution: {e}")
            raise RuntimeError(f"Failed to execute due to: {e}") from e

        return messages

    def setup_agents(self) -> List[RoleBasedAgentModel]:
        """Configures agents based on the provided configuration."""
        logging.info("Setting up agents")
        rag_chain = setup_rag_chain(
            files_path_list=getattr(self.file_config, "files", None),
            url=self.url,
            llm=self.llm,
        )
        rag_tool = self.rag_tool_factory(rag_chain=rag_chain)
        agents: List[RoleBasedAgentModel] = self.agent_factory(
            self.llm, [rag_tool], self.agent_config["roles"]
        )
        return agents

    def create_supervisor(self) -> Any:
        """Creates a supervisor agent configured with specific system prompts and member roles."""
        team_supervisor = self.supervisor_factory(
            self.llm,
            self.agent_config["members"],
            self.agent_config["supervisor_prompts"],
        )
        logging.info("Supervisor agent created")
        return team_supervisor

    def visualise_graph(self):
        """Generates and returns a visual representation of the graph."""
        # Ensure the graph is generated or accessed correctly
        graph_representation = self.graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )
        # Open the image using the appropriate library (assuming the image is stored in a byte stream)
        graph_image = Image.open(BytesIO(graph_representation))
        return graph_image
