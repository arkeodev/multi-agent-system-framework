import logging
from typing import Any, Callable, Dict, Tuple

import streamlit as st
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser

from core.workflow_agent import WorkflowAgent


class EventManager:
    def __init__(self, llm: BaseLLM):
        self.listeners: Dict[str, Callable] = {}
        self.llm = llm
        self.output_parser = StrOutputParser()
        self.workflow_agent = WorkflowAgent(llm)

    def on(self, event: str, callback: Callable):
        self.listeners[event] = callback

    def emit(self, event: str, user_input: Dict[str, Any]) -> Tuple[str, str]:
        if event in self.listeners:
            return self.listeners[event](user_input)
        else:
            logging.warning(f"No listener for event: {event}")
            return "No handler for this event.", "WAITING_FOR_QUERY"

    def handle_query(self, user_input: Dict[str, Any]) -> Tuple[str, str]:
        logging.info(f"Handling query event: {user_input}")
        logging.info(f"Conversation history: {st.session_state.conversation_history}")
        return self.workflow_agent.process_query(
            user_input, st.session_state.conversation_history
        )

    def handle_confirm_query(self, user_input):
        logging.info(f"Handling confirm_query event: {user_input}")
        return "Query confirmed. Let's move on to document input.", "DOCUMENT_INPUT"

    def handle_document_input(self, user_input):
        logging.info(f"Handling document_input event: {user_input}")
        return "Document received. Let's move on to URL input.", "URL_INPUT"

    def handle_url_input(self, user_input):
        logging.info(f"Handling url_input event: {user_input}")
        return "URL received. Let's move on to API input.", "API_INPUT"

    def handle_api_input(self, user_input):
        logging.info(f"Handling api_input event: {user_input}")
        return (
            "API details received. Let's move on to database input.",
            "DATABASE_INPUT",
        )

    def handle_database_input(self, user_input):
        logging.info(f"Handling database_input event: {user_input}")
        return "Database details received. Let's generate agents.", "GENERATE_AGENTS"

    def handle_generate_agents(self, user_input):
        logging.info(f"Handling generate_agents event: {user_input}")
        return "Agents generated. Ready to run the graph.", "RUN_GRAPH"

    def handle_run_graph(self, user_input):
        logging.info(f"Handling run_graph event: {user_input}")
        return (
            "Graph execution complete. What would you like to do next?",
            "WAITING_FOR_QUERY",
        )
