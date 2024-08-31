import logging
from enum import Enum
from typing import Any, Dict

import streamlit as st

from core.event_manager import EventManager


class ChatState(Enum):
    WAITING_FOR_QUERY = "WAITING_FOR_QUERY"
    CONFIRMING_QUERY = "CONFIRMING_QUERY"
    DOCUMENT_INPUT = "DOCUMENT_INPUT"
    URL_INPUT = "URL_INPUT"
    API_INPUT = "API_INPUT"
    DATABASE_INPUT = "DATABASE_INPUT"
    GENERATE_AGENTS = "GENERATE_AGENTS"
    RUN_GRAPH = "RUN_GRAPH"


class InputType(Enum):
    QUERY = "query"
    COMMAND = "command"
    DOCUMENT = "document"
    URL = "url"
    API = "api"
    DATABASE = "database"


class UserInput:
    def __init__(self, type: InputType, content: str):
        self.type = type
        self.content = content


class StateManager:
    def __init__(self, event_emitter: EventManager):
        self.event_emitter = event_emitter
        if "messages" not in st.session_state:
            st.session_state.conversation_history = []
        self.state = ChatState.WAITING_FOR_QUERY
        self.subscribe_to_events()

    def subscribe_to_events(self):
        events = [
            "query",
            "confirm_query",
            "document_input",
            "url_input",
            "api_input",
            "database_input",
            "generate_agents",
            "run_graph",
        ]
        for event in events:
            self.event_emitter.on(event, getattr(self.event_emitter, f"handle_{event}"))
            logging.info(f"Subscribed to event: {event}")

    def handle_user_input(self, user_input: str):
        processed_input = self.process_user_input(user_input)
        event = self.determine_event()
        logging.info(f"Emitting event: {event}")
        response, new_state = self.emit_event(event, processed_input)
        self.update_state(new_state)
        self.update_conversation_history(user_input, response)
        self.update_display()
        return response

    def process_user_input(self, prompt: str) -> Dict[str, Any]:
        logging.info(f"Processing user input in state: {self.state}")
        if prompt.startswith("/"):
            logging.info(f"Processing command: {prompt}")
            return {"content": prompt, "type": "human"}
        else:
            logging.info(f"Processing query: {prompt}")
            return {"content": prompt, "type": "human"}

    def determine_event(self) -> str:
        event_map = {
            ChatState.WAITING_FOR_QUERY: "query",
            ChatState.CONFIRMING_QUERY: "confirm_query",
            ChatState.DOCUMENT_INPUT: "document_input",
            ChatState.URL_INPUT: "url_input",
            ChatState.API_INPUT: "api_input",
            ChatState.DATABASE_INPUT: "database_input",
            ChatState.GENERATE_AGENTS: "generate_agents",
            ChatState.RUN_GRAPH: "run_graph",
        }
        return event_map.get(self.state, "query")

    def emit_event(self, event: str, user_input: Dict[str, Any]):
        return self.event_emitter.emit(event, user_input)

    def update_state(self, new_state: str):
        logging.info(f"Updating state to: {new_state}")
        st.session_state.chat_state = new_state

    def update_conversation_history(self, user_input: str, ai_response: str):
        logging.info(
            f"Updating conversation history with user input: {user_input} and AI response: {ai_response}"
        )
        st.session_state.conversation_history.append(
            {"role": "user", "content": user_input}
        )
        st.session_state.conversation_history.append(
            {"role": "assistant", "content": ai_response}
        )

    def update_display(self):
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def get_conversation_history(self):
        return st.session_state.conversation_history
