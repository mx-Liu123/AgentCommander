# File Management & Structure

AgentCommander adopts a **"File-First"** architecture. Unlike systems that hide data in databases, we enforce a strict, human-readable directory structure.

## The B.L.S Structure

Experiments are organized hierarchically to make manual navigation intuitive.

```text
Project_Root/
├── Branch1/              # Major conceptual direction (e.g., "CNN Model")
│   ├── exp1.1.1/         # The Seed Experiment
│   ├── exp1.2.1/         # Level 2 (Successor of 1.1.1)
│   └── exp1.2.2/         # Step 2 (Retry of 1.2.1 after failure)
└── Branch2/              # New direction (e.g., "Transformer Model")
    └── exp2.1.1/
```

### Naming Convention
*   **Branch (B)**: Distinct lineages of evolution.
*   **Level (L)**: Evolutionary depth. Increments on success.
*   **Step (S)**: Trial count within a level. Increments on failure.

## Integrated File Explorer

The **Explorer Tab** in the UI allows you to navigate through the B.L.S structure and inspect the contents of any experiment folder directly.

### Exploring Experiment Artifacts
By clicking into an experiment directory (e.g., `exp1.2.1`), you can access several critical artifacts:

*   **Evaluation Logs** (`eval_out.txt`): View the raw output of your evaluation script to see performance metrics or debug runtime errors.
*   **AI Records** (`history.json`): This file contains the complete cognitive record of the agent for that experiment, including its hypothesis, experiment design, and result analysis.
*   **Strategy Code** (`strategy.py`): Inspect the actual code generated or modified by the agent.
*   **Visual Results** (`*.png`): View plots and charts generated during evaluation (e.g., `best_result.png`) to perform your own visual verification.

### AI Integration in Explorer
*   **Context-Aware Chat**: An embedded AI assistant that treats the currently open folder as its working directory. You can ask questions like "Summarize the error logs in this folder."
*   **Read-Only Mode**: A safety toggle for the AI Chat. When enabled, the AI can read and analyze files but is **strictly forbidden from modifying or creating files**.
*   **Quick Actions**: Standard file operations including copying paths, renaming, or deleting files manually.


