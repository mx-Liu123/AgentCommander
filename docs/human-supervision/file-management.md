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

The **Explorer Tab** in the UI offers more than just viewing files:

*   **Context-Aware Chat**: Ask the AI questions about the *current folder* (e.g., "Summarize the logs here").
*   **Quick Actions**: Copy paths, rename files, or creating new prototypes manually.
*   **Read-Only Mode**: Toggle this to safely inspect running experiments without accidental edits.
