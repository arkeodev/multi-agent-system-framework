import io
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import streamlit as st

from agents.rag import get_documents
from config.config import AgentConfig
from core.app import App
from interfaces.generate_agents import generate_config_json


class Command(ABC):
    def __init__(self, context: Dict[str, Any]):
        self.context = context

    @abstractmethod
    def execute(self):
        pass

    def check_input(self) -> bool:
        if not self.context["file_upload_config"] and not self.context["url"]:
            with st.chat_message("assistant"):
                st.warning("Please provide either a file or URL.")
                self.context["messages"].append(
                    {
                        "role": "assistant",
                        "content": "Please provide either a file or URL.",
                    }
                )
                logging.error("No file or URL provided.")
            return False
        return True

    def verify_config(self) -> bool:
        if not self.context.get("config_json"):
            with st.chat_message("assistant"):
                st.warning("Please generate a configuration first.")
                self.context["messages"].append(
                    {
                        "role": "assistant",
                        "content": "Please generate a configuration first.",
                    }
                )
            logging.error("No configuration provided.")
            return False
        return True


class HelpCommand(Command):
    def execute(self):
        help_text = """
        Available commands:

        **• /help**
        *Display this help message*

        **• /generate-agents**
        *Generate agent configuration*

        **• /run**
        *Run the current configuration*

        **• /visualise**
        *Visualize the graph*

        **• /change-config**
        *Change the current configuration (provide JSON as argument)*
        """
        with st.chat_message("assistant"):
            st.markdown(help_text)
            self.context["messages"].append({"role": "assistant", "content": help_text})


class GenerateAgentsCommand(Command):
    def execute(self):
        if not self.check_input():
            return
        documents = get_documents(
            getattr(self.context["file_upload_config"], "files", None),
            self.context["url"],
        )
        generated_config = generate_config_json(self.context["llm"], documents)
        if not generated_config:
            with st.chat_message("assistant"):
                st.error("Failed to generate configuration.")
                self.context["messages"].append(
                    {
                        "role": "assistant",
                        "content": "Failed to generate configuration.",
                    }
                )
            return
        # Parse the generated config
        config_dict = json.loads(generated_config)
        # Update the config
        self.context["config_json"] = json.dumps(config_dict, indent=2)
        # Display the generated config
        with st.chat_message("assistant"):
            st.code(
                "Configuration generated successfully. Please provide a scenario to execute."
            )
            st.code(self.context["config_json"], language="json")
            self.context["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Configuration generated successfully. Here's the generated configuration:\n\n```json\n{self.context['config_json']}\n```\nPlease provide a scenario to execute.",
                }
            )


class RunConfigCommand(Command):
    def execute(self):
        if not self.check_input() or not self.verify_config():
            return
        # Initialize scenario in context if not present
        if "scenario" not in self.context:
            self.context["scenario"] = ""
        # Prompt for scenario with on_change callback
        st.text_area(
            "Enter the scenario to execute:",
            key="scenario_input",
            value=self.context["scenario"],
            on_change=self._update_scenario,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.button("Accept", on_click=self._run_scenario)
        with col2:
            st.button("Cancel", on_click=self._handle_cancel)

    def _update_scenario(self):
        self.context["scenario"] = st.session_state.scenario_input

    def _run_scenario(self):
        logging.info(f"Running scenario: {self.context["scenario"]}")

        message_placeholder = st.empty()
        try:
            config_dict = json.loads(self.context["config_json"])
            config_dict["scenario"] = self.context["scenario"]
            user_config = AgentConfig.model_validate(config_dict)
            app = App(
                llm=self.context["llm"],
                recursion_limit=self.context["recursion_limit"],
                agent_config=user_config.model_dump(),
                file_config=self.context["file_upload_config"],
                url=self.context["url"],
                langfuse_handler=self.context["langfuse_handler"],
            )
            messages = app.execute_graph(message_placeholder)
            with st.chat_message("assistant"):
                st.markdown("Execution completed. Results:")
                st.code("\n".join(messages))
                self.context["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Execution completed. Results:\n```\n{messages}\n```",
                    }
                )
            st.session_state.scenario = self.context["scenario"]
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error running the config: {str(e)}")
                self.context["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Error running the config: {str(e)}",
                    }
                )

    def _handle_cancel(self):
        self.context["scenario"] = ""
        st.info("Changes discarded.")
        self.context["messages"].append(
            {
                "role": "assistant",
                "content": "Changes discarded. Please provide a scenario to execute.",
            }
        )


class VisualizeGraphCommand(Command):
    def execute(self):
        if not self.verify_config():
            return
        # Check if scenario exists in context, if not, prompt for it
        if not st.session_state.scenario:
            st.warning("Please enter a scenario before visualizing the graph.")
            return

        try:
            logging.info(f"Visualizing graph for config: {self.context['config_json']}")
            config_dict = json.loads(self.context["config_json"])
            config_dict["scenario"] = st.session_state.scenario
            user_config = AgentConfig.model_validate(config_dict)
            logging.debug(f"Visualizing graph for config: {user_config}")
            app = App(
                llm=self.context["llm"],
                recursion_limit=self.context["recursion_limit"],
                agent_config=user_config.model_dump(),
                file_config=self.context["file_upload_config"],
                url=self.context["url"],
                langfuse_handler=self.context["langfuse_handler"],
            )
            graph_image = app.visualise_graph()
            img_byte_arr = io.BytesIO()
            graph_image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()

            with st.chat_message("assistant"):
                st.image(graph_image, caption="Graph Visualization")
                st.download_button(
                    label="Download Graph Image",
                    data=img_byte_arr,
                    file_name="graph_visualization.png",
                    mime="image/png",
                )
                self.context["messages"].append(
                    {
                        "role": "assistant",
                        "content": "Graph visualization generated. You can download the image using the button above.",
                    }
                )
                logging.info(
                    f"Graph visualization generated for scenario: {self.context['scenario']}"
                )
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error visualizing the graph: {str(e)}")
                self.context["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Error visualizing the graph: {str(e)}",
                    }
                )


class ChangeConfigCommand(Command):
    def execute(self):
        if "edited_config" not in self.context:
            self.context["edited_config"] = self.context["config_json"]

        with st.chat_message("assistant"):
            st.markdown("Current configuration:")
            st.text_area(
                "Edit configuration:",
                value=self.context["edited_config"],
                height=300,
                key="config_editor",
                on_change=self._update_edited_config,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.button("Accept", on_click=self._handle_accept)
            with col2:
                st.button("Cancel", on_click=self._handle_cancel)

    def _update_edited_config(self):
        self.context["edited_config"] = st.session_state.config_editor

    def _handle_accept(self):
        try:
            new_config = json.loads(self.context["edited_config"])
            self.context["config_json"] = json.dumps(new_config, indent=2)
            st.success("Configuration updated successfully.")
            self.context["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Configuration updated successfully. New configuration:\n\n```json\n{self.context['config_json']}\n```",
                }
            )
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please provide a valid JSON configuration.")
        except Exception as e:
            st.error(f"Error updating configuration: {str(e)}")

    def _handle_cancel(self):
        self.context["edited_config"] = self.context["config_json"]
        st.info("Changes discarded.")
        self.context["messages"].append(
            {
                "role": "assistant",
                "content": f"Changes discarded. Current configuration:\n\n```json\n{self.context['config_json']}\n```",
            }
        )


class CommandFactory:
    @staticmethod
    def create_command(command: str, context: Dict[str, Any]) -> Command:
        command_map = {
            "/help": HelpCommand,
            "/generate-agents": GenerateAgentsCommand,
            "/run": RunConfigCommand,
            "/visualise": VisualizeGraphCommand,
            "/change-config": ChangeConfigCommand,
        }
        if command in command_map:
            return command_map[command](context)
        return None


def handle_command(command: str, context: Dict[str, Any]):
    cmd = CommandFactory.create_command(command, context)
    if cmd:
        cmd.execute()
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f"Unknown command: {command}. Type /help for available commands."
            )
            context["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Unknown command: {command}. Type /help for available commands.",
                }
            )


# Main function to be called from the Streamlit app
def process_command(command: str, session_state: Dict[str, Any]):
    context = {
        "messages": session_state.messages,
        "file_upload_config": session_state.file_upload_config,
        "url": session_state.url,
        "llm": session_state.llm,
        "config_json": session_state.get("config_json"),
        "recursion_limit": session_state.recursion_limit,
        "langfuse_handler": session_state.langfuse_handler,
        "scenario": session_state.get("scenario", ""),
    }
    handle_command(command, context)

    # Update session state after command execution
    session_state.messages = context["messages"]
    if "config_json" in context:
        session_state.config_json = context["config_json"]
    session_state.scenario = context.get("scenario", "")
