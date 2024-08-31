# workflow_agent.py

from typing import Any, Dict, List, Tuple

from langchain_core.language_models import BaseLLM
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser


class WorkflowAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.output_parser = StrOutputParser()

    def process_query(
        self, user_input: Dict[str, Any], conversation_history: List
    ) -> Tuple[str, str]:
        prompt = self._generate_prompt(user_input, conversation_history)
        response = self.llm.invoke(prompt)
        parsed_response = self.output_parser.parse(response).content
        next_state = self._determine_next_state(parsed_response)
        return parsed_response, next_state

    def _generate_prompt(
        self, user_input: Dict[str, Any], conversation_history: List
    ) -> str:
        # Similar to the existing _format_conversation_history method
        formatted_history = self._format_conversation_history(conversation_history)

        return f"""
        You are an AI assistant tasked with understanding and clarifying user queries. 
        The user has just provided the following input: "{user_input['content']}"
        
        Previous conversation:
        {formatted_history}
        
        Please analyze this input and respond in a natural, conversational tone. Your response should:
        1. Acknowledge the user's query
        2. Provide a brief interpretation of what you understand the query to be about
        3. Ask for any necessary clarifications if the query is ambiguous or lacks detail
        4. Suggest the next step in processing this query (e.g., confirming understanding, requesting more information, or moving to document input)
        
        Remember to keep your response concise and friendly.
        """

    def _determine_next_state(self, response: str) -> str:
        # Logic to determine the next state based on the AI's response
        if "confirm" in response.lower():
            return "CONFIRMING_QUERY"
        elif any(
            keyword in response.lower() for keyword in ["document", "file", "upload"]
        ):
            return "DOCUMENT_INPUT"
        elif "url" in response.lower():
            return "URL_INPUT"
        elif "api" in response.lower():
            return "API_INPUT"
        elif "database" in response.lower():
            return "DATABASE_INPUT"
        else:
            return "CONFIRMING_QUERY"

    def _format_conversation_history(self, history: List) -> str:
        formatted_history = ""
        for message in history[-10:]:  # Only include the last 5 messages
            role = "Human" if isinstance(message, HumanMessage) else "AI"
            formatted_history += f"{role}: {message}\n"
        return formatted_history

    # Add methods for handling specific stages of the workflow
    # e.g., handle_document_input, handle_url_input, etc.
