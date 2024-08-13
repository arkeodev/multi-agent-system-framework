# graph.py

import logging
from functools import partial
from typing import Any, Dict, List, TypedDict

from langchain.agents import AgentExecutor
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """Represents the state of an agent within the graph system, containing messages and tracking the next node."""

    messages: List[str]  # List of messages exchanged with the agent
    next: str  # The identifier of the next agent or node to process


def agent_node(
    state: Dict[str, Any], agent: AgentExecutor, name: str
) -> Dict[str, Any]:
    """Node function for agents to process input and produce output."""
    result = agent.invoke({"messages": state["messages"]})
    logging.debug(f"{name} output: {result['output']}")
    new_messages = state["messages"] + [f"{name}: {result['output']}"]
    return {
        "messages": new_messages,
        "next": "supervisor",  # Transition control back to the supervisor
    }


def supervisor_node(state: Dict[str, Any], supervisor_agent: Any) -> Dict[str, Any]:
    """Process input from the supervisor to determine the next action."""
    # Simulate a decision from the supervisor or replace with an actual API call or logic
    supervisor_decision = supervisor_agent({"messages": state["messages"]})

    # Update the state based on the supervisor's decision
    state["next"] = supervisor_decision.get("next")
    logging.debug(f"Supervisor decision: {supervisor_decision.get("next")}")

    return state


def create_graph(
    agent_dict: Dict[str, AgentExecutor], supervisor_agent: Any
) -> StateGraph:
    """Constructs a state graph dynamically based on configured agent roles and transitions."""
    logging.info("Creating state graph...")
    graph = StateGraph(state_schema=AgentState)

    # Add all agents and the supervisor to the graph
    for name, agent in agent_dict.items():
        graph.add_node(name, partial(agent_node, agent=agent, name=name))
    graph.add_node(
        "supervisor", partial(supervisor_node, supervisor_agent=supervisor_agent)
    )

    # Set up edges from each agent to the supervisor and conditional edges based on supervisor decisions
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
