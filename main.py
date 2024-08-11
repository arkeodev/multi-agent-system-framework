# main.py

import os
from typing import List

import streamlit as st
from langchain_openai import ChatOpenAI

from modules.agent import AgentModel, create_agents
from modules.execution import run_scenario
from modules.rag import load_vectorstore, setup_rag_chain
from modules.supervisor import create_team_supervisor  # Make sure it's updated
from modules.tools import HandbookTool
from modules.utils import (
    load_agent_config,
    load_configuration,
    set_api_keys,
    setup_logging,
)

setup_logging()

# Load configuration
config = load_configuration()
agent_config = load_agent_config()

# Streamlit UI for OpenAI API key and model selection
st.title("Multi-Agent System for Search and Rescue Mission")
set_api_keys()

# Streamlit UI for selecting the model
model_choice = st.selectbox("Select the model to use:", config["models"], index=0)

# Initialize LLM with the selected model
llm = ChatOpenAI(model=model_choice)

# Streamlit UI for selecting the document loader
document_loader_choice = st.selectbox(
    "Select the document loader:", list(config["document_loaders"].keys()), index=0
)

# Define the file path for the vector index
vector_index_path = "vector_index.faiss"

# Scenario text input
scenario = st.text_area("Scenario Text", agent_config["scenario"], height=300)

# Textbox for intermediate steps
steps_box = st.empty()

# Button to trigger the scenario
if st.button("Run Scenario"):
    if os.path.exists(vector_index_path):
        handbook_rag_chain = load_vectorstore(vector_index_path, llm)
    else:
        handbook_rag_chain = setup_rag_chain(
            config["document_urls"][document_loader_choice],
            llm,
            vector_index_path,
        )
    handbook_tool = HandbookTool(handbook_rag_chain=handbook_rag_chain)
    agents: List[AgentModel] = create_agents(llm=llm, tools=[handbook_tool])
    # Change to use only RAG chain and remove manual member inputs
    candc_agent = create_team_supervisor(
        llm=llm,
        system_prompt=agent_config["supervisor_prompts"]["initial"],
        members=agent_config["members"],
    )
    agent_dict = {agent.role_name: agent.agent for agent in agents}
    messages = run_scenario(scenario, agent_dict, candc_agent)
    steps_box.text("\n".join(messages))  # Update steps_box with the collected messages
