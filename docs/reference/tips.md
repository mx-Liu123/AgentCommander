# Tips & Best Practices

## Optimizing ML Parameter Search

For a good balance between speed and cost, a duration of **20-30 minutes** for an experiment cycle (including parameter searching) is recommended. This allows sufficient exploration without excessive expenditure.

## Prompt Templating

Use Jinja2-like syntax (e.g., `{{ variable_name }}`) in `llm_generate` nodes to dynamically inject values from the shared context.

## Debugging with CLI

Use `gemini -r`, `qwen -c`, or `opencode -c` inside an experiment directory to inspect the exact prompt sent to the LLM and resume the conversation manually if needed.

## CLI Hygiene

Set your global CLI history file (e.g., `~/gemini/GEMINI.md`) to **Read-Only** to prevent context pollution between different projects.
