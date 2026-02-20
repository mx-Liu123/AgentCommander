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

The **Explorer Tab** in the UI offers a deep integration between your data and AI:

*   **Context-Aware Chat**: An embedded AI assistant that treats the currently open folder as its working directory. You can ask questions like "Summarize the error logs in this folder."
*   **Read-Only Mode**: A safety toggle for the AI Chat. When enabled, the AI can read and analyze files but is **strictly forbidden from modifying or creating files**. This is ideal for safe inspection of running experiments.
*   **Quick Actions**: Standard file operations including copying paths, renaming, or deleting files manually.

