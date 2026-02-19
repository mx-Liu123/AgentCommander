# CLI Reference & Tips

AgentCommander relies on the underlying CLI tools for model inference. Optimizing their configuration can significantly improve performance and stability.

## Gemini CLI

*   **Installation**: `npm install -g @google/gemini-cli@latest`
*   **Authentication**: Run `gemini login` to authenticate with your Google account.

### 🚀 Enabling Preview Models
The standard models (e.g., `gemini-pro`) are stable but may lag behind in reasoning capability. For complex coding tasks, we recommend **Gemini 1.5 Pro** or **Flash 2.0**.

To access them:
1.  Ensure you have the latest CLI version.
2.  When configuring the Agent in the UI, type the model alias manually if it doesn't appear in the dropdown (e.g., `gemini-1.5-pro-latest`).

### 🛡️ Context Isolation (Important)
By default, the CLI maintains a history file (`~/.gemini/GEMINI.md`). If multiple projects write to this file, context pollution can occur.

**Tip**: The AgentCommander UI automatically handles this isolation by setting the working directory as the context root, but for global CLI usage, consider setting your global `GEMINI.md` to **Read-Only** to prevent accidental contamination from manual CLI usage.

## Qwen CLI

*   **Installation**: `npm install -g @qwen/cli`

### 🆓 Free Tier (OAuth)
Qwen offers an "OpenAI-compatible, OAuth free tier" which provides ~2,000 free requests/day.
1.  Run `qwen login`.
2.  Follow the OAuth flow.
3.  This is excellent for long-running evolutionary tasks where token costs on other platforms might be prohibitive.

## OpenCode AI

An open-source oriented backend.

*   **Installation**: `npm install -g opencode-ai`
*   **Configuration**: Run `opencode login` to authenticate.

## Claude Code (Anthropic)

Integration with Anthropic's official CLI.

*   **Installation**: `npm install -g @anthropic-ai/claude-code`
*   **Configuration**:
    1.  Run `claude login`.
    2.  This will open a browser window to authenticate with your Anthropic Console account.
    3.  Grant permission to the CLI.

## Common Issues

### "Model not found"
*   **Cause**: Your CLI version is outdated.
*   **Fix**: Run `npm update -g @google/gemini-cli` (or `@qwen/cli`).

### Node.js Warnings
*   If you see warnings about experimental fetch APIs, upgrade Node.js to the latest LTS (Long Term Support) version.
