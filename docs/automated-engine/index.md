# Agent Capability

AgentCommander empowers AI agents to mimic the cognitive loop of a human researcher: **Hypothesize, Observe, Reference, and Summarize**.

By orchestrating these cognitive steps into a structured workflow, we achieve a highly efficient, automated trial-and-error engine.

## The Default Research Cycle

In the default configuration (`default_graph.json`), every time a new experiment folder is created, the Agent follows a rigorous scientific process:

### 1. Hypothesis Generation
*   **Context**: The agent reads the current code, previous execution logs, and critically, performs **Visual Analysis** on any result plots (e.g., loss curves, prediction scatter plots).
*   **Action**: Based on this multimodal input, it formulates a specific hypothesis for the current experiment (e.g., "The model is overfitting, so I will increase dropout").

### 2. Implementation & Debugging
*   **Coding**: The agent modifies the code (`strategy.py`) to test the hypothesis.
*   **Observability**: It proactively adds debug print statements to gather more information during execution, just like a human developer would.
*   **Execution**: The code is run against the immutable `evaluator.py`.

### 3. Summary & History
*   **Record**: Regardless of success or failure, the agent summarizes the experiment design and performance into `history.json`.
*   **Evolution**: If the score improves, it advances to the next level. If it fails, it retries until a limit is reached.

### 4. Meta-Reflection (Getting Unstuck)
*   **Memory**: When starting a new experiment, the agent reads the history of previous attempts.
*   **External Inspiration**: If the system detects a stagnation (multiple generations without breakthrough), it triggers a special **Meta-Analysis** step. The agent uses **Online Search** tools to find relevant papers (Arxiv) or open-source repositories (GitHub) to find fresh inspiration.