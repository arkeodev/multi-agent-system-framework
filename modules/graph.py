# graph.py
import logging
from functools import partial
from typing import Any, Dict, List, TypedDict
from langchain.agents import AgentExecutor
from langgraph.graph import END, StateGraph

class PilotTeamState(TypedDict):
    messages: List[str]
    next: str

def agent_node(
    state: PilotTeamState, agent: AgentExecutor, name: str
) -> Dict[str, Any]:
    """Invoke the agent and return the response."""
    # Invoke the agent with the current state and log the response
    result = agent.invoke({"messages": state["messages"]})
    logging.info(f"{name} response: {result['output']}")
    return {
        "messages": state["messages"] + [f"{name}: {result['output']}"],
        "next": "candc",
    }

def create_graph(
    pilot_agent: AgentExecutor,
    copilot_agent: AgentExecutor,
    cso_agent: AgentExecutor,
    candc_agent: Any,
) -> StateGraph:
    """Create the conversation flow graph."""
    # Define nodes for each agent
    pilot_node = partial(agent_node, agent=pilot_agent, name="Pilot")
    copilot_node = partial(agent_node, agent=copilot_agent, name="Copilot")
    cso_node = partial(agent_node, agent=cso_agent, name="CSO")

    # Initialize the graph
    candc_graph = StateGraph(PilotTeamState)

    # Add nodes to the graph
    candc_graph.add_node("Pilot", pilot_node)
    candc_graph.add_node("Copilot", copilot_node)
    candc_graph.add_node("CSO", cso_node)
    candc_graph.add_node("candc", candc_agent)

    # Define edges between nodes
    candc_graph.add_edge("Pilot", "candc")
    candc_graph.add_edge("Copilot", "candc")
    candc_graph.add_edge("CSO", "candc")
    candc_graph.add_conditional_edges(
        "candc",
        lambda x: x["next"],
        {"Pilot": "Pilot", "Copilot": "Copilot", "CSO": "CSO", "FINISH": END},
    )

    # Set the entry point of the graph
    candc_graph.set_entry_point("candc")
    logging.info("Graph setup complete")
    return candc_graph
