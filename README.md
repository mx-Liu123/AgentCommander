# AgentCommander

AgentCommander is an advanced, graph-based workflow execution engine designed to orchestrate AI Agents for complex, iterative tasks. It provides a visual interface for designing workflows, managing experiments, and dynamically configuring agent behaviors.

## Key Features

*   **Visual Workflow Editor**: Design complex agent loops and decision trees using a node-based interface.
*   **LLM Integration**: Seamlessly integrate with Large Language Models (LLMs) like Gemini and Claude for decision making and code generation.
*   **Experiment Management**: Automatically track and visualize experiment history, metrics, and branches.
*   **Dynamic Configuration**: Manage global variables and system settings through a centralized UI with support for multiple configuration profiles.
*   **Robust Execution**: Includes features for session recovery, error handling, and parallel execution.
*   **Web UI**: A modern, responsive web interface for real-time monitoring and control.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/AgentCommander.git
    cd AgentCommander
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    conda create -n agent_commander python=3.10
    conda activate agent_commander
    ```
    *OR*
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

1.  **Start the UI Server:**
    ```bash
    bash run_ui.sh
    ```
    This will start the web server (default port 8080) and open the UI.

2.  **Access the UI:**
    Open your browser and navigate to `http://localhost:8080`.

3.  **Load Configuration:**
    *   The system will automatically create a default `config.json` if one is missing.
    *   You can load example configurations or create your own in the "Control Panel".

4.  **Run an Example Task:**
    *   Ensure the `Root Dir` in the Control Panel points to `example/diabetes_sklearn` (relative path).
    *   Click "Start Agent" to begin the automated experiment loop.
    *   Monitor progress in the "Console" and "Explorer" tabs.

## Configuration

The `config.json` file controls the core behavior of the agent system. You can manage this file directly via the UI's "Control Panel".

*   **root_dir**: The working directory where experiments and data are stored.
*   **n_cycles**: The number of experiment iterations to run.
*   **global_vars**: Variables available to the agent (e.g., system prompts, python paths).
*   **llm_changeable_vars**: List of variables the LLM is allowed to modify in `history.json`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
