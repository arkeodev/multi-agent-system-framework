# supervisor.py

import json
import logging
from typing import Any, Callable, Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_team_supervisor(llm: Any, agent_config: Dict) -> Callable:
    """Create a supervisor agent to manage the team dynamically based on a specified prompt and team members."""
    # List of options that the supervisor can choose from, including the possibility to end the session
    options = ["FINISH"] + agent_config["members"]

    # Function definition for routing between agents
    function_def = {
        "name": "route",
        "description": "Select the next role to act.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}],
                },
            },
            "required": ["next"],
        },
    }

    # Define the prompt and create the supervisor agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                agent_config["supervisor_prompts"]["initial"],
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                agent_config["supervisor_prompts"]["decision"],
            ),
        ]
    ).partial(options=str(options))

    # Bind the prompt with the language model and function definition
    supervisor_agent = prompt | llm.bind_functions(
        functions=[function_def], function_call="route"
    )

    def invoke_supervisor(state):
        """Invokes the supervisor agent with the current state and updates the state based on the decision made by the supervisor."""
        result = supervisor_agent.invoke(state)
        logging.debug(f"Supervisor response: {result}")
        if (
            hasattr(result, "additional_kwargs")
            and "function_call" in result.additional_kwargs
        ):
            function_call = result.additional_kwargs["function_call"]
            if function_call and function_call.get("name") == "route":
                arguments = json.loads(function_call["arguments"])
                state["next"] = arguments["next"]
        return state

    return invoke_supervisor
