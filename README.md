# AgentCommander

miaoxin.liu@u.nus.edu

As an NUS Astrophysics PhD student, I created AgentCommander out of a deep personal need to streamline the iterative process of machine learning model development and debugging. My motivation stems from the belief that the exhaustive trial-and-error inherent in ML research should not be a burden for human developers. Instead, these repetitive, yet crucial, tasks are ideally suited for automation, freeing up human intelligence to focus on higher-level creative pursuits, systemic design, and conceptual exploration. AgentCommander is my answer to this, aiming to automate the tedious parts of ML iteration so researchers can dedicate their energy to innovation.

AgentCommander is an advanced, graph-based workflow execution engine designed to orchestrate AI Agents for complex, iterative tasks. Built on top of the **Gemini CLI**, it empowers Machine Learning engineers and researchers to build highly customizable, infinite-loop workflows for tasks like **symbolic regression**, **hyperparameter optimization**, and **autonomous model refinement**.

![Control Panel](control_panel.png)

## Key Features

*   **Visual Workflow Editor with AI Assistant**: Design complex agent loops and decision trees using a node-based interface, enhanced by a built-in AI Assistant that can modify the workflow directly via natural language commands (e.g., "Add a shell check after step 2").
![Workflow Editor](workflow_editor.png)
*   **Gemini CLI Integration**: Deeply integrated with the Gemini ecosystem for powerful, prompt-driven code generation and analysis.
*   **Infinite Iteration & Advanced Learning**: Create self-improving loops where the agent experiments, learns from failures, and refines its strategy indefinitely. Advanced features like the **"Lesson" mechanism** (to learn from past errors) and **online search integration** (for inspiration) are available in example workflows to boost continuous improvement.
*   **ML & Symbolic Regression**: specifically tailored to assist in discovering mathematical formulas and optimizing ML models through iterative experimentation.
*   **Experiment Management & Evolutionary Tree**: Automatically track and visualize experiment history, metrics, and branches as an **"evolutionary tree"**, where each experiment node connects to its parent.
*   **Dynamic Configuration**: Manage global variables and system settings through a centralized UI.

## Installation

### Prerequisites
1.  **Gemini CLI**: Ensure you have the Gemini CLI installed and configured.
    *   *See official Gemini CLI documentation for installation instructions.*

### Gemini CLI Configuration Recommendation
To leverage the latest capabilities, including the powerful Pro3 and Flash3 models, it is highly recommended to enable "Gemini Preview" in your Gemini CLI settings. This allows AgentCommander to access cutting-edge model versions.

*   **Note for Students**: Gemini currently offers a one-year free Pro user trial for student accounts. This is a great opportunity to explore the full potential of the latest Gemini models.

2.  **Python 3.10+**

### Steps
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

## Reference Experiment Structure

To help you get started, the `example/diabetes_sklearn/` directory provides a reference implementation for organizing your machine learning experiments. This structure is designed to be robust and prevent "cheating" by the LLM.

### 1. File Organization
*   **Root Evaluation Script (`evaluator.py`)**: Placed at the project root (e.g., `example/diabetes_sklearn/evaluator.py`), this file serves as the ground truth for assessment. It should be treated as **immutable** by the LLM. By keeping it outside the experiment's working directory, we prevent the agent from modifying the evaluation logic to artificially inflate scores.
*   **Seed Experiment Directory (`Branch_example/exp_example/`)**: This folder acts as the "seed" for your evolutionary tree. It contains the initial `strategy.py` (the model or logic to be improved) and any auxiliary files required for execution.
*   **Experiment History (`history.json`)**: Within each experiment directory, `history.json` automatically stores all calculated metrics and the optimal results. The workflow can conveniently write key-value pairs into this file using the `write_history` module.

### 2. Execution Flow
In the default workflow, the agent evaluates a strategy by running a shell command similar to this:

```bash
cd {current_exp_path} && \
export PYTHONPATH=$PYTHONPATH:../../ && \
{venv} -c "from evaluator import evaluate; print('Best metric:', evaluate('strategy.py'))" > eval_out.txt 2>&1; \
cat eval_out.txt
```

*   **Isolation**: The command changes directory to the specific experiment folder (`{current_exp_path}`).
*   **Context**: It adds the project root to `PYTHONPATH` so the local `strategy.py` can be imported by the root `evaluator.py`.
*   **Output**: All generated files (metrics, logs, artifacts) remain contained within the experiment directory, ensuring a clean workspace for every iteration.

### 3. Hyperparameter Search Pattern
The provided example also demonstrates a pattern for hyperparameter optimization:
*   **`strategy.py`**: Defines both the model logic and the parameter space to be searched.
*   **`evaluator.py`**: Imports the strategy, reads the parameter space, and executes the search (e.g., Grid Search, Bayesian Optimization), returning the best metric found.

### 4. Evolutionary Progress Logic (B, L, S)
The example workflow uses a Branch (B), Level (L), Step (S) logic to manage evolutionary progress:
*   **Branch (B)**: A new branch is typically initiated manually, often when a significant breakthrough or new conceptual direction is explored.
*   **Level (L)**: Represents a progression in depth, advanced when an experiment shows significant improvement or a breakthrough.
*   **Step (S)**: Denotes different explorations within the same level when no breakthrough is achieved. Each new step is based on the optimal result from the previous (L-1) level, allowing for iterative refinement.

This mechanism is implemented within the example workflow and is fully modifiable by the user to suit their specific progress tracking needs.

### Note on Customization
Because the AgentCommander workflow is graph-based and highly customizable, this structure is merely a **recommendation**. You are free to modify the folder hierarchy, execution commands, and evaluation logic to fit the specific needs of your project.

## Configuration

The `config.json` file controls the core behavior of the agent system. You can manage this file directly via the UI's "Control Panel".

*   **root_dir**: The working directory where experiments and data are stored.
*   **n_cycles**: The number of experiment iterations to run.
*   **global_vars**: These are hardcoded environment variables (e.g., API keys, system paths) accessible throughout the workflow.
*   **llm_changeable_vars**: A list of variable names that the LLM is explicitly allowed to modify during runtime. These often represent hyperparameters or learned parameters.
*   **workflow**: The workflow definition graph. Can be a full JSON object or a string path to a separate JSON file (e.g., `"my_workflows/workflow_v1.json"`).
*   **Shared Context**: All nodes in the workflow exchange data through a shared variable pool, known as the `context`. This allows for flexible data flow between different modules without strict parameter passing.

## Tips & Best Practices

*   **Optimizing ML Parameter Search**: For a good balance between model iteration speed and computational cost, a duration of **20-30 minutes** for an `exp` directory coupled with ML parameter searching is often a cost-effective approach. This allows sufficient exploration without excessive expenditure.
*   **Prompt Templating**: Within `llm_generate` nodes, you can use Jinja2-like templating syntax (e.g., `{{ variable_name }}`) to dynamically inject values from the shared context into your prompts. This is crucial for creating adaptive and data-driven agents.
*   **Debugging Prompts with `gemini -r`**: After running your workflow, you can use the `gemini -r` command in the corresponding experiment directory (`cd {exp_path} && gemini -r`) to restore the latest conversation. This allows you to inspect the exact prompt that was sent to the LLM, verifying that your templating and context variables are working as expected.

## Security Considerations & Disclaimer

While AgentCommander leverages the Gemini CLI, which primarily operates within its designated working directory, it's crucial to understand the inherent risks:

*   **Gemini Model Access (Default YOLO Mode)**: By default, when invoked by AgentCommander, Gemini models operate with the `-y` (YOLO - You Only Live Once) parameter. This means the model is granted permission to use *any* available tool and has *arbitrary privileges* within the working directory. This design choice enables powerful automation but requires extreme caution.
*   **File Access Scope**: The Gemini CLI typically focuses on files within the specified working directory (`root_dir`). However, any generated scripts or commands executed by the agent **theoretically could interact with or modify files outside this directory**, especially if the agent's logic or your system's configuration allows it.
*   **Best Practices**: For maximum security, it is highly recommended to:
    *   **Use a Sandboxed Environment**: Run AgentCommander and its agents within a container (e.g., Docker) or a virtual machine.
    *   **Restrict User Permissions**: Execute the application and agents with a user account that has minimal necessary file system permissions, preventing unintended modifications to critical system files or sensitive data.
*   **Disclaimer**: This project is provided "as is," without warranty of any kind, express or implied. The developers are not responsible for any damage, loss of data, or security breaches resulting from the use or misuse of this software. Users are solely responsible for ensuring the secure operation of their environment and for validating any code or actions generated by the AI agents.

## Todo

*   [ ] **Claude Code Support**: Integrate Anthropic's Claude as an alternative LLM backend.
*   [ ] **Parallel Workflow Example**: Add concrete examples and templates for running experiments in parallel.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
