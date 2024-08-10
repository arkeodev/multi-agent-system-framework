# execution.py
import logging
from typing import Any, List
from langgraph.graph.state import CompiledStateGraph
from modules.graph import StateGraph, create_graph

def run_scenario(
    scenario: str,
    pilot_agent: Any,
    copilot_agent: Any,
    cso_agent: Any,
    candc_agent: Any,
) -> List[str]:
    """Run the search and rescue scenario."""
    state = {"messages": [scenario], "next": "candc"}
    candc_graph: StateGraph = create_graph(
        pilot_agent, copilot_agent, cso_agent, candc_agent
    )
    compiled_graph: CompiledStateGraph = candc_graph.compile()

    message_list = []  # Initialize message list with the scenario description

    while True:
        logging.info(f"Invoking chain with current state: {state}")
        result = compiled_graph.invoke(input=state)

        # Always update the state and messages before checking for the finish condition
        if 'messages' in result:
            message_list.extend(result['messages'])  # Append new messages to the list
            state.update(result)
            logging.info("\n".join(result["messages"]))

        if "FINISH" in result.get("next", ""):
            logging.info("Scenario finished")
            break  # Exit after processing all updates for the current iteration

        logging.info("---")

    return message_list  # Return the accumulated messages
