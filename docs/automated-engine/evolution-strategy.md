# Evolutionary Strategy

AgentCommander manages the lifecycle of experiments using a structured evolution logic.

## Default Strategy: B/L/S (Branch/Level/Step)

The default workflow (`default_graph.json`) implements a heuristic evolutionary tree:

*   **Branch (B)**: Represents major conceptual directions or distinct lineages.
*   **Level (L)**: Represents depth of optimization. When an experiment succeeds (improves the metric), the system spawns a new generation at **Level + 1**, inheriting the successful traits.
*   **Step (S)**: Represents horizontal trial-and-error. When an experiment fails, the system retries at **Step + 1** with a modified hypothesis, keeping the Level constant.

## Customization & Parallelism

It is important to note that B/L/S is simply the **default template**.

### Flexible Graph Architecture
The entire logic is defined in the visual workflow editor. You are free to redesign this graph to implement:
*   **Genetic Algorithms**: Selection, Crossover, Mutation nodes.
*   **Bayesian Optimization**: Logic nodes that update hyperparameters based on past results.

### Parallel Exploration
The underlying engine supports **Parallel Execution**. You can design workflows where multiple agents explore different branches simultaneously, or where a "Manager Agent" spawns multiple "Worker Agents" to solve sub-problems in parallel. This scalability allows AgentCommander to leverage massive compute resources for accelerated discovery.