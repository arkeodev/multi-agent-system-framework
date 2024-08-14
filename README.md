# Multi-Agent System Framework

This project implements a multi-agent system framework using Streamlit, LangChain and LangGraph designed to facilitate complex tasks by coordinating multiple agents. The framework is highly modular, allowing for easy integration and swapping of components such as models, document loaders, and agent roles based on specific scenarios.

## Features

- **Modular Architecture**: Swap out AI models, document loaders, and other components effortlessly.
- **Dynamic Agent Configuration**: Configure agent roles dynamically through a JSON configuration, adapting the application for various scenarios without code changes.
- **Interactive UI**: Use Streamlit to provide an intuitive interface for inputting scenarios and viewing results.
- **Configurable Scenarios**: Run custom scenarios by configuring agents and tasks through a user-friendly JSON interface.

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

## Configuration

Modify the `config.json` file for general settings like models and document loaders, and use `agent_configuration.json` to define agent roles and scenario-specific behaviors dynamically:

### config.json
```json
{
  "models": ["gpt-4o", "gpt-4o-mini"],
  "embedding_model": "text-embedding-3-small",
  "document_loaders": {
    "pdf": "PyMuPDFLoader",
    "txt": "SimpleTextLoader"
  }
}
```

### agent_configuration.json
Here's an example of configuring a scenario for a disaster response team:
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
    "scenario_source_documents": ["https://arxiv.org/pdf/2408.00026"],
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

## License

Distributed under the MIT License. See `LICENSE` for more information.
