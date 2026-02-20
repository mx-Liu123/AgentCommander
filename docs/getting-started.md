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

## Next Steps

Now that the server is running, you have two paths:

1.  **Initialize a New Project**: Use the Auto-Setup Wizard to scaffold your environment.
    *   👉 Go to **"Protocols & Setup" > "Auto-Setup Wizard"**.
2.  **Run a Demo**: Try the included Diabetes example to see the agent in action immediately.
    *   👉 Go to **"Guides" > "Diabetes Example"**.

For advanced configuration (e.g., enabling Gemini Pro), check **"Reference & Tips" > "CLI Guide"**.

