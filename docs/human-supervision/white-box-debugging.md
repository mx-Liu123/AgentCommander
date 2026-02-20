# White-box Debugging

AgentCommander adopts a "White-box" philosophy. Unlike frameworks that hide agent logic inside opaque execution environments, we allow you to "step into" the agent's mind at any time.

## 💡 Pro Tip: Resuming Context

Since each experiment node runs in its own isolated directory, you can leverage the persistence features of the CLI tools for deep debugging.

### Why is this powerful?
*   **Inspect Final Prompts**: See exactly what the CLI received after all template variables (like `{{hint}}` or `{{parent_metric}}`) were resolved. This is crucial for verifying your prompt engineering logic.
*   **Review History**: Access the full conversation log (e.g., `GEMINI.md`) to understand *why* the agent made a specific decision or error.
*   **Manual Intervention**: If an agent gets stuck in a loop, you can intervene manually and then let the workflow continue.

### How to do it (Gemini Example):
1.  **Open Terminal**.
2.  **Navigate** to the experiment folder (e.g., `cd my_project/Branch1/exp1.2.1`).
3.  **Run `gemini -r`** (Resume mode).
4.  **Interact**:
    *   **Debug Prompt**: Type *"Show me the last prompt you received."* to verify the input context.
    *   **Analyze Logic**: Ask *"Why did you choose this architecture?"*
    *   **Manual Fix**: Directly instruct *"Fix the syntax error on line 45"* to help it recover.

*For **Qwen CLI** or **OpenCode AI** users, use `qwen -c` or `opencode -c` to achieve the same effect.*
