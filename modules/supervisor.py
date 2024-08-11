# supervisor.py
import json
import logging
from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


def create_team_supervisor(
    llm: ChatOpenAI, system_prompt: str, members: List[str]
) -> Any:
    """Create a supervisor agent to manage the team."""
    # Define the options for the supervisor to choose the next role
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    # Define the prompt and create the supervisor agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Who should act next? Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options))
    supervisor_agent = prompt | llm.bind_functions(
        functions=[function_def], function_call="route"
    )

    def invoke_supervisor(state):
        result = supervisor_agent.invoke(state)
        logging.info(f"Supervisor response: {result}")
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
