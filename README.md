# Multi-agent Omni LangGraph Executer (m o l e)

<div style="text-align: center;">
  <img src="images/mole.png" alt="MOLE" width="150" height="150">
</div>

This project implements a multi-agent system framework using Streamlit, LangChain, and LangGraph designed to facilitate complex tasks by coordinating multiple agents. The framework is highly modular, allowing for easy integration and swapping of components such as models, document loaders, and agent roles based on specific scenarios.

## Features

- **Modular Architecture**: Swap out AI models, document loaders, and other components effortlessly.
- **Dynamic Agent Configuration**: Configure agent roles dynamically through a JSON configuration, adapting the application for various scenarios without code changes.
- **Supervisor Managed Orchestration**: All the management is done by a supervisor agent.
- **Interactive UI**: Use Streamlit to provide an intuitive interface for inputting scenarios and viewing results.
- **Configurable Scenarios**: Run custom scenarios by configuring agents and tasks through a user-friendly JSON interface.
- **Document Processing**: Load and process various document types, and convert them into LangChain's `Document` format with metadata.
- **Web Scraping**: Scrape websites using SmartGraph, with support for exclusion patterns and maximum page depth to refine the scraping process.
- **FAISS Integration**: Store processed documents in a FAISS vector store, enabling efficient retrieval for complex queries.

## Prerequisites

Before running the application, ensure you have the following:
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

Modify the `agent_configuration.json` to define agent roles and scenario-specific behaviors dynamically.

### Sample agent_configuration.json
Here's an example of configuring a scenario for a disaster response team. You can download the scenario file of this configuration file from [this link](https://static.e-publishing.af.mil/production/1/af_a1/publication/afh1/afh1.pdf).
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
Navigate to the provided URL by Streamlit in your web browser to interact with the application.

## Modules

- **main.py**: The entry point of the application that handles the UI and scenario execution.
- **agent.py**: Defines the agents and their specific roles.
- **rag.py**: Handles document loading and the setup of Retrieval-Augmented Generation chains.
- **supervisor.py**: Manages the coordination among multiple agents.
- **execution.py**: Orchestrates the running of scenarios using the defined agents.
- **tools.py**: Contains utility tools used by agents and the RAG chain for document retrieval and processing.
- **document_loader.py**: Handles loading of documents based on file type, including custom loaders for unsupported types.
- **smart_graph_handler.py**: Integrates with SmartGraph for scraping data from web URLs, supporting exclusion patterns and page depth constraints.
- **graph.py**: Constructs and manages the state graph used to coordinate agent interactions dynamically within a scenario.


## License

Distributed under the MIT License. See `LICENSE` for more information.