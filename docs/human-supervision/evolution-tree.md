# Evolutionary Tree Visualization

Tracking hundreds of experiments is impossible with just a console log. AgentCommander provides a real-time **Progress Tree** to visualize the lineage of your agents.

## The Progress Tree

Located in the **Control Panel**, this graph dynamically updates as the agent explores.

### Visual Guide
*   **Nodes**: Each node represents a unique experiment folder (e.g., `exp1.2.1`).
*   **Edges**: Directed lines show the inheritance path (Parent $\to$ Child).
*   **Color Coding**:
    *   🟢 **Green**: Success (Metric improved). Spawns a new Level.
    *   🔴 **Red**: Failure (Metric worsened or crashed). Spawns a retry Step.
    *   🔵 **Blue**: Active (Currently running).

## Interaction
*   **Hover**: See quick metrics (Score, Hypothesis summary).
*   **Click**: (Planned Feature) Navigate directly to the file explorer for that experiment.
