# execution.py
import logging
from typing import Any, Dict, List

from langchain.agents import AgentExecutor

from modules.graph import create_graph


def run_scenario(
    scenario: str, agent_dict: Dict[str, AgentExecutor], supervisor_agent: Any
) -> List[str]:
    state = {"messages": [scenario], "next": "supervisor"}
    compiled_graph = create_graph(agent_dict, supervisor_agent).compile()
    message_list = []
    while True:
        logging.info(f"Current state before invoke: {state}")
        result = compiled_graph.invoke(input=state)
        logging.info(f"Result after invoke: {result}")
        if result.get("messages"):
            message_list.extend(result["messages"])
        if result.get("next") != state["next"]:
            state["next"] = result["next"]
        if "FINISH" in state.get("next", ""):
            break
    return message_list
