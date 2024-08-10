# main.py
import os
import streamlit as st
from langchain_openai import ChatOpenAI

from modules.agent import create_agents
from modules.execution import run_scenario
from modules.rag import setup_rag_chain, load_vectorstore
from modules.supervisor import create_team_supervisor
from modules.utils import set_api_keys, setup_logging

setup_logging()

# Streamlit UI for OpenAI API key input
st.title("Multi-Agent System for Search and Rescue Mission")

set_api_keys()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o")

# Define the file path for the vector index
vector_index_path = "vector_index.faiss"

# Scenario
scenario = """
Mission Brief: Operation Ocean Guardian

A cargo ship, the SS Meridian, has been missing for 72 hours in the North Atlantic. Your mission is to locate the ship using a P-8 Poseidon aircraft. The crew consists of a Pilot, Copilot, and Combat Systems Operator (CSO).

Objectives:
1. Navigate to the last known coordinates of the SS Meridian.
2. Search for the ship using the aircraft's sensors.
3. Communicate findings and coordinate with search and rescue teams.

Execute the mission, with each crew member performing their role.
"""

st.write("### Scenario")
scenario_text = st.text_area("Scenario Text", scenario, height=300, label="")

# Textbox for intermediate steps
steps_box = st.empty()

def initialize_agents_and_supervisor():
    global pilot_agent, copilot_agent, cso_agent, candc_agent
    pilot_agent, copilot_agent, cso_agent = create_agents(llm, handbook_rag_chain)
    candc_agent = create_team_supervisor(
        llm,
        "You are Control and Command managing an aircraft crew: Pilot, Copilot, CSO. Direct who acts next or FINISH when done.",
        ["Pilot", "Copilot", "CSO"],
    )

# Button to trigger the scenario
if st.button("Run Scenario"):
    if os.path.exists(vector_index_path):
        # Load existing vector store
        handbook_rag_chain = load_vectorstore(vector_index_path, llm)
    else:
        # Set up RAG chains and save the vector store
        handbook_rag_chain = setup_rag_chain(
            "https://static.e-publishing.af.mil/production/1/af_a1/publication/afh1/afh1.pdf",
            "pilot_demo_airforce_handbook",
            llm,
            vector_index_path
        )

    initialize_agents_and_supervisor()
    messages = run_scenario(
        scenario_text, pilot_agent, copilot_agent, cso_agent, candc_agent
    )
    steps_box.text("\n".join(messages))  # Update steps_box with the collected messages
    