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
    "initial": "You are the supervisor of a disaster response team. Direct the team's actions or decide to conclude the operations.",
    "decision": "Who should act next? Or should we conclude operations? Select one of: {options}"
  },
  "members": [
    "Commander",
    "Logistics Officer",
    "Field Agent"
  ],
  "roles": [
    {
      "name": "Commander",
      "prompt": "You are the Commander. Your role is to oversee the operation and make strategic decisions."
    },
    {
      "name": "Logistics Officer",
      "prompt": "You are the Logistics Officer. Manage resources and ensure all logistical needs are met."
    },
    {
      "name": "Field Agent",
      "prompt": "You are the Field Agent. Assess the situation on the ground and provide updates to the team."
    }
  ],
  "document_urls": [
    "https://example.com/emergency_protocol.pdf"
  ],
  "scenario": "Disaster Response Scenario: A major earthquake has hit the urban area of Metropolis. Buildings are damaged, and there are numerous injuries. The team's mission includes: \n1. Assessment of the situation on the ground. \n2. Coordination of rescue and medical teams. \n3. Distribution of supplies and resources.\nExecute the mission with each team member performing their designated roles."
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
