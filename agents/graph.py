# graph.py

import logging
from functools import partial
from typing import Any, Dict, List, TypedDict

from langchain.agents import AgentExecutor
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from config.config import AGENT_SUPERVISOR


class AgentState(TypedDict):
    messages: List[str]
    next: str
    scratchpad: List[Dict[str, Any]]
    step: int


def serialize_scratchpad(scratchpad: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Serializes the scratchpad contents into a list of dictionaries."""
    serialized_scratchpad = []
    for message in scratchpad:
        try:
            serialized_scratchpad.append(
                message.dict()
                if hasattr(message, "dict")
                else {"content": str(message)}
            )
        except Exception as e:
            logging.warning(
                f"Failed to serialize scratchpad item: {message}, error: {e}"
            )
            serialized_scratchpad.append({"content": "unserializable object"})
    return serialized_scratchpad


def update_scratchpad(state: AgentState, agent_name: str, output: str) -> AgentState:
    """Updates the state with the latest agent interaction."""
    step_info = {"step": state["step"], "agent": agent_name, "output": output}
    state["scratchpad"].append(step_info)
    if agent_name == AGENT_SUPERVISOR:
        state["step"] += 1
    return state


def agent_node(state: AgentState, agent: AgentExecutor, name: str) -> AgentState:
    """Processes a node in the graph representing an agent."""
    logging.info(f"Agent Node {name} - Current Step: {state['step']}")
    dynamic_context = {
        "messages": state["messages"],
        "scratchpad": state["scratchpad"][-1] if state["scratchpad"] else None,
        "step": state["step"],
    }
    result = agent.invoke(dynamic_context)
    # Format the message to include the step number and make the agent name look like a markdown heading
    new_message = f"# Step {state['step']} - {name}\n{result['output']}"
    if not state["messages"] or state["messages"][-1] != new_message:
        state["messages"].append(new_message)

    state = update_scratchpad(state, name, result["output"])
    state["next"] = AGENT_SUPERVISOR
    return state


def supervisor_node(state: AgentState, supervisor_agent: Any) -> AgentState:
    """Directs the graph's flow based on the supervisor's decision."""
    logging.info(f"Supervisor Node - Current Step: {state.get('step', 'Not Set')}")
    dynamic_context = {
        "messages": state["messages"],
        "scratchpad": state["scratchpad"][-1] if state["scratchpad"] else None,
        "step": state["step"],
    }
    supervisor_decision = supervisor_agent(dynamic_context)
    selected_agent = supervisor_decision.get("next")
    state["next"] = selected_agent
    scratchpad_entry = f"Step {state['step']}: Supervisor selected {selected_agent}."
    state = update_scratchpad(state, AGENT_SUPERVISOR, scratchpad_entry)
    logging.info(f"Supervisor decision: {selected_agent}")
    return state


def create_graph(
    agent_dict: Dict[str, AgentExecutor], supervisor_agent: Any
) -> StateGraph:
    """Constructs a state graph dynamically based on configured agent roles and transitions."""
    logging.info("Creating state graph...")
    graph = StateGraph(state_schema=AgentState)
    for name, agent in agent_dict.items():
        graph.add_node(name, partial(agent_node, agent=agent, name=name))
    graph.add_node(
        AGENT_SUPERVISOR, partial(supervisor_node, supervisor_agent=supervisor_agent)
    )
    for name in agent_dict:
        graph.add_edge(name, AGENT_SUPERVISOR)
    graph.add_conditional_edges(
        AGENT_SUPERVISOR,
        lambda state: state["next"],
        {name: name for name in agent_dict} | {"FINISH": END},
    )
    graph.set_entry_point(AGENT_SUPERVISOR)
    logging.info("Graph setup complete")
    return graph
