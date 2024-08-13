# execution.py

import logging
from typing import Any, Dict, List

from langchain.agents import AgentExecutor
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphRecursionError

from modules.graph import create_graph


def execute_scenario(
    scenario: str,
    agent_dict: Dict[str, AgentExecutor],
    supervisor_agent: Any,
    recursion_limit: int,
) -> List[str]:
    """Executes the scenario within a constructed graph, handling agent interactions and supervisor decisions."""
    # Initial state setup with the first message being the scenario description.
    state = {"messages": [scenario], "next": "supervisor"}

    # Configuration for the runnable graph, including recursion limit.
    config = RunnableConfig(recursion_limit=recursion_limit)

    # Compile the graph from the agent dictionary and supervisor.
    app = create_graph(agent_dict, supervisor_agent).compile()

    logging.info(f"Current state before invoke: {state}")
    result = None

    try:
        # Stream outputs from the graph execution.
        for output in app.stream(input=state, config=config):
            for key, value in output.items():
                logging.info(f"Node '{key}': {value}")
                result = value
    except GraphRecursionError:
        # Log error if the recursion limit is reached during graph execution.
        logging.error(
            "Graph recursion limit reached, adjust the limit in the configuration if needed."
        )

    logging.info(f"Result after invoke: {result}")

    message_list = []
    # Aggregate messages from the results.
    if result and "messages" in result:
        message_list.extend(result["messages"])

    # Update the state based on the result from the graph.
    if result and "next" in result and result["next"] != state["next"]:
        state["next"] = result["next"]

    return message_list
