# Multi-agent Omni LangGraph Executer (m o l e)

<div style="text-align: center;">
  <img src="images/mole.png" alt="MOLE" width="150" height="150">
</div>

This project implements a robust multi-agent system framework leveraging Streamlit, LangChain, and LangGraph to manage and execute complex tasks by coordinating multiple agents. The framework is designed with modularity in mind, allowing for seamless integration and replacement of components such as AI models, document loaders, and agent roles tailored to specific scenarios.

## Features

- **Modular Architecture**: Easily swap AI models, document loaders, and other components to fit your specific needs.
- **Agent-Based System**: Leverage multiple agents, each with distinct roles, to collaboratively achieve complex objectives.
- **Dynamic Agent Configuration with Scenario Generator**: Utilize an integrated language model to dynamically generate agent roles and configuration scenarios.
- **Supervisor-Managed Agent Orchestration**: A supervisor agent manages the orchestration of tasks, ensuring efficient collaboration among agents.
- **Interactive UI**: Streamlit-based interface for an intuitive experience in defining scenarios and visualizing results.
- **Customizable Scenarios**: Easily configure agents and their tasks using a user-friendly JSON interface, enabling the execution of tailored scenarios.
- **Comprehensive Document Processing**: Load and process a variety of document types, converting them into LangChain's `Document` format enriched with metadata.
- **FAISS Integration**: Store and retrieve processed documents efficiently using FAISS vector storage, enabling rapid responses to complex queries.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.12
- Poetry for dependency management and environment setup

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arkeodev/multi-agent-system-framework.git
   cd multi-agent-system-framework
   ```

2. **Install dependencies and set up the virtual environment using Poetry:**
   ```bash
   poetry install
   ```

## Agent Configuration

For those who wish to manually create a scenario configuration file, can modify the sample provided below. This sample outlines how to define agent roles and tailor scenarios dynamically.

### Sample agent_configuration.json

Here’s an example of configuring a scenario for an astronomy research team. The scenario can be downloaded from [this link](https://arxiv.org/pdf/2408.00026).

```json
{
    "supervisor_prompts": {
        "initial": "You are the Mission Coordinator managing an X-ray astronomy crew. Your task is to oversee the study of the Virgo Cluster using the LEIA X-ray imager. Direct who acts next or FINISH when done.",
        "decision": "Who should act next? Or should we FINISH? Choose from: {options}"
    },
    "members": [
        "Data Analyst",
        "Observation Specialist",
        "Imaging Scientist"
    ],
    "roles": [
        {
            "name": "Data Analyst",
            "prompt": "You are a Data Analyst. Your role is to process and analyze the telemetry and spectral data received from the LEIA observations. Interpret findings in relation to the known characteristics of the Virgo Cluster."
        },
        {
            "name": "Observation Specialist",
            "prompt": "You are the Observation Specialist. Manage the observation schedules and ensure the telescope's alignment and calibration are optimized for capturing high-quality X-ray images of the Virgo Cluster."
        },
        {
            "name": "Imaging Scientist",
            "prompt": "You are an Imaging Scientist. Your task is to process the images captured by LEIA, correct for any distortions or anomalies, and prepare the data for detailed analysis by the Data Analyst."
        }
    ],
    "scenario": "Mission Brief: Virgo Cluster X-ray Study\nThe mission involves using the LEIA X-ray imager to conduct detailed observations of the Virgo Cluster. Your team consists of a Data Analyst, Observation Specialist, and Imaging Scientist.\nObjectives:\n1. Align and calibrate the LEIA telescope to begin observations of the Virgo Cluster.\n2. Capture and process X-ray images, identifying key features and anomalies.\n3. Analyze the spectral and imaging data to provide insights into the cluster's composition and dynamics.\nExecute the mission, with each crew member performing their specific roles."
}
```

## Usage

To start the application, activate the Poetry environment and run:

```bash
poetry run streamlit run main.py
```

Navigate to the URL provided by Streamlit in your web browser to interact with the application.

## Modules

- **main.py**: The main entry point of the application, handling the UI and scenario execution.
- **app.py**: Manages the application’s core functionality, including agent setup and scenario execution.
- **agent.py**: Defines the agents, detailing their roles and responsibilities within a scenario.
- **rag.py**: Manages document loading and sets up Retrieval-Augmented Generation (RAG) chains for efficient document processing.
- **supervisor.py**: Coordinates multiple agents by managing their tasks and ensuring smooth execution.
- **execution.py**: Orchestrates the execution of scenarios, enabling agents to work together seamlessly.
- **tools.py**: Provides utility tools for agents and the RAG chain, assisting in document retrieval and processing.
- **document_loader.py**: Loads documents based on their file type, including custom loaders for unsupported types.
- **scenario_generator.py**: Automatically generates scenario configuration files in JSON format using a provided input file or URL.
- **graph.py**: Constructs and manages the state graph used to dynamically coordinate agent interactions within a scenario.

## License

Distributed under the MIT License. See `LICENSE` for more information.