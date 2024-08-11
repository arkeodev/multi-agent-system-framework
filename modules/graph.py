# graph.py
import logging
from functools import partial
from typing import Any, Dict, List, TypedDict

from langchain.agents import AgentExecutor
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """Represents the state of an agent in the system."""

    messages: List[str]  # List of messages exchanged with the agent
    next: str  # The identifier of the next agent or state


def agent_node(
    state: Dict[str, Any], agent: AgentExecutor, name: str
) -> Dict[str, Any]:
    """Invoke the agent and return the response, ensuring state updates."""
    result = agent.invoke({"messages": state["messages"]})
    new_messages = state["messages"] + [f"{name}: {result['output']}"]
    return {
        "messages": new_messages,
        "next": "supervisor",  # Ensure there's always an update to 'next'
    }


def supervisor_node(state: Dict[str, Any], supervisor_input: str) -> Dict[str, Any]:
    """Process input from the supervisor to determine the next action."""
    if supervisor_input == "FINISH":
        state["next"] = "END"  # This ends the graph processing
    else:
        state["next"] = supervisor_input  # Set next agent to act
    return state


def create_graph(
    agent_dict: Dict[str, AgentExecutor], supervisor_agent: Any
) -> StateGraph:
    """Create a state graph dynamically based on configured agent roles and transitions."""
    logging.info("Creating state graph...")
    graph = StateGraph(state_schema=AgentState)
    for name, agent in agent_dict.items():
        graph.add_node(name, partial(agent_node, agent=agent, name=name))
    graph.add_node("supervisor", supervisor_agent)

    for name in agent_dict:
        graph.add_edge(name, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {name: name for name in agent_dict} | {"FINISH": END},
    )

    graph.set_entry_point("supervisor")
    logging.info("Graph setup complete")
    return graph