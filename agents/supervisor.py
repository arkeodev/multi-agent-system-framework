# supervisor.py

import json
import logging
from typing import Any, Callable, Dict, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.agents import agent_registry
from config.config import FINISH, ROUTE_NAME


def create_team_supervisor(
    llm: Any, members: List[str], supervisor_prompts: Dict[str, str]
) -> Callable:
    """Create a supervisor for the team."""
    # Add standard agent names to the members list
    all_members = members + agent_registry.get_all_names()
    options = [FINISH] + all_members
    logging.info(f"Supervisor options: {options}")

    function_def = {
        "name": ROUTE_NAME,
        "description": "Select the next role to act.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {"title": "Next", "anyOf": [{"enum": options}]},
            },
            "required": ["next"],
        },
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", supervisor_prompts["initial"]),
            MessagesPlaceholder(variable_name="messages"),
            ("system", supervisor_prompts["decision"]),
        ]
    ).partial(options=str(options))

    supervisor_agent = prompt | llm.bind_functions(
        functions=[function_def], function_call="route"
    )

    def invoke_supervisor(state: Dict) -> Dict:
        logging.info("Invoking supervisor")
        dynamic_context = {
            "messages": state["messages"],
            "scratchpad": state.get("scratchpad", []),
            "step": state["step"],
        }
        result = supervisor_agent.invoke(dynamic_context)
        if (
            hasattr(result, "additional_kwargs")
            and "function_call" in result.additional_kwargs
        ):
            function_call = result.additional_kwargs["function_call"]
            if function_call and function_call.get("name") == ROUTE_NAME:
                arguments = json.loads(function_call["arguments"])
                state["next"] = arguments["next"]
                logging.info(f"Supervisor decided next agent: {state['next']}")
        return state

    return invoke_supervisor
