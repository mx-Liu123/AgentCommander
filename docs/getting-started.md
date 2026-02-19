# Getting Started

## Prerequisites

1.  **Python 3.10+**
2.  **LLM CLI Tool** (One of the following):
    *   [Gemini CLI](https://github.com/google/gemini-cli) (Recommended): `npm install -g @google/gemini-cli@latest`
        *   *Tip*: Enable "Gemini Preview" in settings to access Pro 1.5/Flash 2.0 models.
    *   [Qwen CLI](https://github.com/qwen-cli): `npm install -g @qwen/cli`
        *   *Tip*: Qwen offers a free tier via OAuth.

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

Once the UI is running, you can:
1.  Go to **"Reference & Tips" > "CLI Guide"** to configure your LLM backend.
2.  Use the **Auto-Setup Wizard** to create your first experiment environment.
