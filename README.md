# Multi-Agent System Framework

This project implements a multi-agent system framework using Streamlit and LangChain, designed to facilitate complex tasks by coordinating multiple agents. The framework is highly modular, allowing easy integration and swapping of components such as models, document loaders, and agent roles based on specific scenarios.

## Features

- **Modular Architecture**: Easily swap out AI models, document loaders, and other components.
- **Dynamic Agent Roles**: Configure and extend agent roles dynamically through a user-friendly interface.
- **Interactive UI**: Leverage Streamlit to provide an intuitive interface for inputting scenarios and viewing results.
- **Configurable Scenarios**: Run custom scenarios by configuring agents and tasks through the UI.

## Prerequisites

Before you can run the application, you will need to have the following installed:
- Python 3.8 or higher
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

3. **Setup environment variables:**
   Set your OpenAI API key in the virtual environment:
   ```bash
   poetry run export OPENAI_API_KEY='your-openai-api-key'
   ```

## Configuration

Modify the `config.json` file to set up different models, document loaders, and default scenarios:
```json
{
  "models": ["gpt-4o", "gpt-3.5-turbo"],
  "document_loaders": ["PDF Loader", "Text Loader"],
  "document_urls": {
    "PDF Loader": "https://example.com/pdf1.pdf",
    "Text Loader": "https://example.com/text1.txt"
  },
  "default_scenario": "Enter your scenario here..."
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
