# execution.py

import logging
from typing import Any, Dict, Generator, Optional

from langchain.agents import AgentExecutor
from langchain_core.runnables.config import RunnableConfig
from langfuse.callback import CallbackHandler
from langgraph.errors import GraphRecursionError

from modules.config.config import AGENT_SUPERVISOR
from modules.graph import AgentState, create_graph


def execute_scenario(
    scenario: str,
    agent_dict: Dict[str, AgentExecutor],
    supervisor_agent: Any,
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
        recursion_limit=recursion_limit, callbacks=[langfuse_handler]
    )
    app = create_graph(agent_dict, supervisor_agent).compile()

    logging.debug(f"Initial state before execution: {initial_state}")
    sent_messages = []

    try:
        for output in app.stream(input=initial_state, config=config):
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
