# Evolutionary Strategy (B/L/S)

AgentCommander organizes experiments in a tree structure:

*   **Branch (B)**: Major conceptual directions (e.g., 'Use CNN', 'Use Transformer').
*   **Level (L)**: Depth of optimization. Moves to L+1 upon success.
*   **Step (S)**: Trials within a level. Moves to S+1 upon failure (retry).

This ensures systematic exploration rather than random walking.
