# commands.py

import io
import json

import streamlit as st

from agents.agent_config import generate_config_json
from agents.rag import get_documents
from config.config import AgentConfig
from core.app import App


def handle_command(command):
    """Handle different commands entered in the chat widget."""
    if command == "/help":
        display_help()
    elif command == "/generate_config":
        generate_config()
    elif command == "/run":
        run_config()
    elif command == "/visualize":
        visualize_graph()
    elif command == "/change_config":
        change_config()
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f"Unknown command: {command}. Type /help for available commands."
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Unknown command: {command}. Type /help for available commands.",
                }
            )


def display_help():
    """Display help information for available commands."""
    help_text = """
    Available commands:

    **• /help**
    *Display this help message*

    **• /generate_config**
    *Generate agent configuration*

    **• /run**
    *Run the current configuration*

    **• /visualize**
    *Visualize the graph*

    **• /change_config**
    *Change the current configuration (provide JSON as argument)*
    """
    with st.chat_message("assistant"):
        st.markdown(help_text)
        st.session_state.messages.append({"role": "assistant", "content": help_text})


def generate_config():
    """Generate a configuration based on uploaded files or URL content."""
    if not check_input():
        return
    documents = get_documents(
        getattr(st.session_state.file_upload_config, "files", None),
        st.session_state.url,
    )
    generated_config = generate_config_json(st.session_state.llm, documents)
    if not generated_config:
        with st.chat_message("assistant"):
            st.error("Failed to generate configuration.")
            st.session_state.messages.append(
                {"role": "assistant", "content": "Failed to generate configuration."}
            )
        return
    st.session_state.config_json = generated_config
    with st.chat_message("assistant"):
        st.code("Configuration generated successfully.")
        st.code(generated_config, language="json")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Configuration generated successfully. Here's the generated configuration:\n\n```json\n{generated_config}\n```",
            }
        )


def run_config():
    """Handle execution."""
    if not check_input() or not verify_config():
        return
    message_placeholder = st.empty()
    try:
        user_config = AgentConfig.model_validate_json(st.session_state.config_json)
        app = App(
            llm=st.session_state.llm,
            recursion_limit=st.session_state.recursion_limit,
            agent_config=user_config.model_dump(),
            file_config=st.session_state.file_upload_config,
            url=st.session_state.url,
            langfuse_handler=st.session_state.langfuse_handler,
        )
        messages = app.execute_graph(message_placeholder)
        with st.chat_message("assistant"):
            st.markdown("Execution completed. Results:")
            st.code("\n".join(messages))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Execution completed. Results:\n```\n{messages}\n```",
                }
            )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error running the config: {str(e)}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error running the config: {str(e)}"}
            )


def visualize_graph():
    """Handle graph visualization."""
    if not verify_config():
        return
    try:
        user_config = AgentConfig.model_validate_json(st.session_state.config_json)
        app = App(
            llm=st.session_state.llm,
            recursion_limit=st.session_state.recursion_limit,
            agent_config=user_config.model_dump(),
            file_config=st.session_state.file_upload_config,
            url=st.session_state.url,
            langfuse_handler=st.session_state.langfuse_handler,
        )
        graph_image = app.visualise_graph()

        # Convert PIL Image to bytes
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
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Graph visualization generated. You can download the image using the button above.",
                }
            )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error visualizing the graph: {str(e)}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Error visualizing the graph: {str(e)}",
                }
            )


def check_input():
    """Check if the input is valid."""
    if not st.session_state.file_upload_config and not st.session_state.url:
        with st.chat_message("assistant"):
            st.warning("Please provide either a file or URL.")
            st.session_state.messages.append(
                {"role": "assistant", "content": "Please provide either a file or URL."}
            )
        return False
    return True


def verify_config():
    """Check if the configuration is valid."""
    if not st.session_state.config_json:
        with st.chat_message("assistant"):
            st.warning("Please generate a configuration first.")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Please generate a configuration first.",
                }
            )
        return False
    return True


def change_config():
    """Open the current configuration for editing."""
    if "edited_config" not in st.session_state:
        st.session_state.edited_config = st.session_state.config_json

    with st.chat_message("assistant"):
        st.markdown("Current configuration:")
        st.text_area(
            "Edit configuration:",
            value=st.session_state.edited_config,
            height=300,
            key="config_editor",
            on_change=update_edited_config,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.button("Accept", on_click=handle_accept)
        with col2:
            st.button("Cancel", on_click=handle_cancel)


def update_edited_config():
    st.session_state.edited_config = st.session_state.config_editor


def handle_accept():
    try:
        # Validate JSON
        new_config = json.loads(st.session_state.edited_config)
        # Update the configuration
        st.session_state.config_json = json.dumps(new_config, indent=2)
        st.success("Configuration updated successfully.")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Configuration updated successfully. New configuration:\n\n```json\n{st.session_state.config_json}\n```",
            }
        )
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please provide a valid JSON configuration.")
    except Exception as e:
        st.error(f"Error updating configuration: {str(e)}")


def handle_cancel():
    st.session_state.edited_config = st.session_state.config_json
    st.info("Changes discarded.")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"Changes discarded. Current configuration:\n\n```json\n{st.session_state.config_json}\n```",
        }
    )