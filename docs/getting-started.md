# Getting Started

## Prerequisites

1.  **Python 3.10+**
2.  **LLM CLI Tool** (One of the following):
    *   [Gemini CLI](https://github.com/google/gemini-cli) (Recommended): `npm install -g @google/gemini-cli@latest`
        *   *Tip*: Enable "Gemini Preview" in settings to access Pro 3/Flash 3 models.
    *   [Qwen CLI](https://github.com/qwen-cli): `npm install -g @qwen/cli`
        *   *Tip*: Qwen offers a free tier via OAuth.
    *   **OpenCode AI**: `npm install -g opencode-ai`
    *   **Claude Code**: `npm install -g @anthropic-ai/claude-code`

## Installation

### OS Support
*   ✅ **Linux & macOS**: Fully supported (native).
*   ⚠️ **Windows**: **Highly recommended to use WSL2**.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/AgentCommander.git
    cd AgentCommander
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start the UI Server:**
    ```bash
    bash run_ui.sh
    ```
    This will start the web server at `http://localhost:8080`.

## Initializing Your First Project (UI Wizard)

The easiest way to start a new project is using the built-in **Experiment Setup** wizard directly in the web UI.

1.  **Navigate to "Setup"**: Click the "Experiment Setup" tab in the UI sidebar.
2.  **Select a Template**:
    *   **`[Case: You only have Dataset]`**: Choose this if you are starting from scratch with just `X.npy` and `Y.npy`.
    *   **`[Case: You have Training Code]`**: Choose this if you want to optimize an existing script (BYOC mode).
3.  **Configure**: Fill in the required fields (e.g., Project Name, Absolute Path to Data).
4.  **Launch**: Click **🚀 Run Setup Script**. The integrated console will show the progress as it scaffolds your environment.

## Quick Test: Running the Diabetes Example

You can verify your installation immediately by running our pre-configured Scikit-Learn example without any setup.

1.  **Open Control Panel**: In the UI, stay on the **Control** tab.
2.  **Set Root Directory**: Set the `Root Directory` field to `example/diabetes_sklearn`.
3.  **Configure `eval_cmd`**:
    *   In the **Global Variables** list, find `eval_cmd`.
    *   Ensure it is set to: `python evaluator.py` (ensure your environment's python is used).
4.  **Save & Start**: Click **💾 Save Config**, then click **▶ Start Agent**.
    *   The agent will immediately start its first cycle, analyzing the diabetes data and generating model hypotheses.

## Next Steps

Once you have a running agent:
1.  Go to **"Reference & Tips" > "CLI Guide"** to fine-tune your LLM backend settings.
2.  Check **"Agent Capability"** to understand how the evolutionary loop works.
