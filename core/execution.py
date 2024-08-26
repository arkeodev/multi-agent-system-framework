# execution.py

import logging
from typing import Generator, Optional

from langchain_core.runnables.config import RunnableConfig
from langfuse.callback import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from agents.graph import AgentState
from config.config import AGENT_SUPERVISOR


def execute_graph(
    graph: CompiledStateGraph,
    scenario: str,
    recursion_limit: int,
    langfuse_handler: Optional[CallbackHandler] = None,
) -> Generator[str, None, None]:
    """Executes a scenario within a constructed graph, handling agent interactions and supervisor decisions.
    Yields:
        str: Messages generated during the scenario execution.
    """
    scenario_message = f"# Step 1 - Scenario\n{scenario}"
    initial_state = AgentState(
        messages=[scenario_message], next=AGENT_SUPERVISOR, scratchpad=[], step=1
    )
    config = RunnableConfig(
        recursion_limit=recursion_limit,
        callbacks=[langfuse_handler] if langfuse_handler else [],
    )
    logging.debug(f"Initial state before execution: {initial_state}")
    sent_messages = []
    try:
        for output in graph.stream(input=initial_state, config=config):
            for key, value in output.items():
                logging.debug(f"Node '{key}' processed with output: {value}")
                if "messages" in value:
                    for message in value["messages"]:
                        if message not in sent_messages:
                            yield message
                            sent_messages.append(message)
    except GraphRecursionError:
        logging.error(
            "Graph recursion limit reached, consider adjusting the limit in the configuration."
        )
