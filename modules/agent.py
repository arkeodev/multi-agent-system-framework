# agent.py
from typing import Any, Tuple

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


def create_agents(
    llm: ChatOpenAI, handbook_rag_chain: Any
) -> Tuple[AgentExecutor, AgentExecutor, AgentExecutor]:
    """Create agents for the pilot, copilot, and CSO."""

    @tool
    def retrieve_information(query: str) -> str:
        """Provides detailed information from the 'Air Force Handbook'."""
        # Add console log
        print(f"Retrieving information for query: {query}")
        return handbook_rag_chain.invoke({"question": query})

    # Create individual agents with their respective roles
    pilot_agent = create_agent(
        llm,
        [retrieve_information],
        "You are a fully qualified pilot. Speak only as your role.",
    )
    copilot_agent = create_agent(
        llm,
        [retrieve_information],
        "You are a fully qualified copilot. Speak only as your role.",
    )
    cso_agent = create_agent(
        llm,
        [retrieve_information],
        "You are a fully qualified Combat Systems Operator. Speak only as your role.",
    )
    return pilot_agent, copilot_agent, cso_agent


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
    """Create an agent with the specified tools and prompt."""
    # Define the prompt and create the agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)
