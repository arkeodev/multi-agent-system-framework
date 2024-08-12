# execution.py
import logging
from typing import Any, Dict, List

from langchain.agents import AgentExecutor
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphRecursionError

from modules.graph import create_graph


def run_scenario(
    scenario: str, agent_dict: Dict[str, AgentExecutor], supervisor_agent: Any
) -> List[str]:
    state = {"messages": [scenario], "next": "supervisor"}
    config = RunnableConfig(recursion_limit=30)
    app = create_graph(agent_dict, supervisor_agent).compile()
    logging.info(f"Current state before invoke: {state}")
    result = None
    try:
        for output in app.stream(input=state, config=config):
            for key, value in output.items():
                logging.info(f"Node '{key}':")
                result = value
    except GraphRecursionError:
        logging.error("Graph recursion limit reached.")
    logging.info(f"Result after invoke: {result}")
    message_list = []
    if result.get("messages"):
        message_list.extend(result["messages"])
    if result.get("next") != state["next"]:
        state["next"] = result["next"]
    return message_list
