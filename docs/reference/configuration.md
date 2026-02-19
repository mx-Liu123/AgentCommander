# Configuration (`config.json`)

The `config.json` file controls the global behavior.

## Key Fields
*   **root_dir**: Working directory.
*   **n_cycles**: Number of iterations.
*   **global_vars**: Variables accessible in prompt templates via `{{ var_name }}`.
    *   `eval_cmd`: The shell command to run evaluation.
    *   `plot_names`: Filenames of plots to analyze.
