# Multi-agent Omni LangGraph Executer (m o l e)

<div style="text-align: center;">
  <img src="images/mole.png" alt="MOLE" width="150" height="150">
</div>

This project implements a robust multi-agent system framework leveraging Streamlit, LangChain, and LangGraph to manage and execute complex tasks by coordinating multiple agents. The framework is designed with modularity in mind, allowing for seamless integration and replacement of components such as AI models, document loaders, and agent roles tailored to specific scenarios.

## Features

- **Modular Architecture**: Easily swap AI models, document loaders, and other components to fit your specific needs.
- **Agent-Based System**: Leverage multiple agents, each with distinct roles, to collaboratively achieve complex objectives.
- **Dynamic Agent Configuration**: Utilize an integrated language model to dynamically generate agent roles and configuration based on input documents or URLs.
- **Supervisor-Managed Agent Orchestration**: A supervisor agent manages the orchestration of tasks, ensuring efficient collaboration among agents.
- **Interactive UI**: Streamlit-based interface for an intuitive experience in defining scenarios and visualizing results.
- **Customizable Scenarios**: Easily input custom scenarios to be executed by the configured agents.
- **Comprehensive Document Processing**: Load and process a variety of document types, converting them into LangChain's `Document` format enriched with metadata.
- **FAISS Integration**: Store and retrieve processed documents efficiently using FAISS vector storage, enabling rapid responses to complex queries.
- **LangFuse Integration**: Seamlessly trace and monitor LLM operations using LangFuse, either through an `.env` file or direct input of keys via the UI.

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

## LangFuse Integration

LangFuse is integrated into the framework to allow detailed tracing and monitoring of the LLM operations. You can configure the integration using either a `.env` file or directly through the UI.

### Option 1: .env File

Ensure your `.env` file includes the following variables:

```env
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=your_host_url
```

### Option 2: UI Input

Alternatively, you can enable LangFuse and input the keys directly through the Streamlit UI:

1. Check the "Enable LangFuse" checkbox.
2. Expand the "LangFuse LLM operation tracing" section.
3. Input your LangFuse Public Key, Secret Key, and Host Name.
4. Optionally, test the connection to ensure it's set up correctly.


## License

Distributed under the MIT License. See `LICENSE` for more information.
