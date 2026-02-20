<div align="center">

# AgentCommander

**Orchestrating AI Agents for Iterative Scientific Research.**

[![Documentation](https://img.shields.io/badge/📖_Docs-Read_Now-green?style=for-the-badge&logo=gitbook)](https://mx-Liu123.github.io/AgentCommander/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

📧 **Contact:** [miaoxin.liu@u.nus.edu](mailto:miaoxin.liu@u.nus.edu) | 📖 **Documentation:** [Read Online](https://mx-Liu123.github.io/AgentCommander/)

</div>

---

## Motivation

Born from the complex computational needs of scientific research, AgentCommander addresses a critical bottleneck in machine learning: the exhaustive cost of manual trial-and-error. 

I attempted to iterate and optimize machine learning code using various existing tools, but found them lacking in flexibility. **Cursor Agent** excels at code completion but cannot design long-term evolutionary paths. **OpenEvolve/AlphaEvolve** offers powerful population-based evolution but focuses on group behavior rather than deep, customized single-agent optimization.

AgentCommander fills this gap. It is built on the belief that repetitive iteration is a task for machines, not humans. By automating the debugging and refinement cycle with a highly customizable graph-based workflow, AgentCommander empowers researchers to focus on high-level creative pursuits and systemic design.

## What is AgentCommander?

![AgentCommander Concept](docs/assets/images/main_pic.png)

**AgentCommander was born from the actual demands of scientific research.**

Refined through rigorous practical application, it is a graph-based workflow engine designed to orchestrate AI Agents for complex, iterative tasks. Built to leverage the diverse ecosystem of **LLM CLIs** (Gemini, Qwen, Claude, etc.), it enables Machine Learning engineers to construct highly customizable, infinite-loop workflows.

Unlike "black-box" agents, AgentCommander prioritizes **Human-Centric Evolution**. You define the search space and evaluation logic; the agent handles the exhaustive execution loop.

## Core Features

*   **Hierarchical Workflows**: Orchestrate macro-level evolutionary strategies (Outer Loop) and micro-level experiment execution (Inner Loop).
*   **White-box Transparency**: Every experiment runs in its own folder. Use `cd` and `gemini -r` (or `qwen -c`) to inspect prompts, review history, or intervene manually.
*   **Visual Editor & Assistant**: Design complex topologies via a drag-and-drop UI or natural language commands.
*   **Multi-Model Support**: Native integration with Gemini, Qwen, Claude, and OpenCode CLIs.
*   **Safety Sandboxing**: Directory-level isolation with filesystem snapshots and automated rollback.
*   **HPC Support**: Built-in templates for PBS/Slurm clusters (e.g., NUS Vanda server).

## Adapt to Your Research in Minutes

The **Auto-Setup Wizard** makes it easy to integrate AgentCommander into your existing workflow without rewriting your code.

![Auto-Setup Wizard](docs/assets/images/auto_setup.png)

1.  **[Case: You only have Dataset]**: Scaffolds a complete project (splitting, strategy, and evaluator) from raw data.
2.  **[Case: You have Training Code]**: Instantly adapts your existing scripts into the agent system by adding a simple interface for weight loading and evaluation.

## Quick Start

1.  **Install Prerequisites**:
    *   **Python 3.10+**
    *   **LLM CLI**: `npm install -g @google/gemini-cli@latest` (or `qwen`, `claude`)
2.  **Clone & Install**:
    ```bash
    git clone https://github.com/mx-Liu123/AgentCommander.git
    cd AgentCommander
    pip install -r requirements.txt
    ```
3.  **Launch**:
    ```bash
    bash run_ui.sh
    ```
    Open `http://localhost:8080`, go to the **Experiment Setup** tab, and scaffold your first project.

---

*Licensed under the Apache License 2.0.*
