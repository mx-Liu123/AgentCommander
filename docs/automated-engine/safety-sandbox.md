# Security & Sandbox System

AgentCommander addresses the safety challenges of autonomous agents with a multi-layer "Soft Sandbox" built on top of the filesystem.

## Directory-Level Sandboxing

Unlike other agent frameworks where file system isolation can be difficult to enforce, the CLI-based approach allows AgentCommander to strictly limit the agent's read/write access to specific experiment directories. This ensures that the agent can freely experiment within its sandbox without risking modifications to your system-level files.

*   **Isolation**: The CLI tools focus on files within the specified working directory (`root_dir`). 
*   **Transparent Debugging (White-box)**: Every experiment runs in its own folder. You can simply `cd` into any experiment directory and run `gemini -r`, `qwen -c`, or `opencode -c` to **resume the conversation**. This allows you to inspect the final formatted prompt, review the history, and manually intervene.

## File Permission Modes (LLM Nodes)

To prevent LLM agents from modifying unauthorized files, AgentCommander implements a strict File Permission System for each `llm_generate` node:

*   **Strict (Read-Only)**: The LLM is strictly forbidden from modifying existing files.
*   **Restricted (Whitelist/Blacklist)**: The LLM can only modify the files or folders explicitly listed (e.g., `strategy.py`).
*   **Open (Allow All)**: Unrestricted access within the working directory.

**Enforcement**: The system creates a filesystem snapshot before execution and compares it after. Any unauthorized changes are immediately **reverted**.

## No-Exec Mechanism (Execution Ban)

To prevent agents from accidentally (or maliciously) executing unfinished or dangerous code during the generation phase, we implement a strict **No-Exec Lock**:

*   **Behavior**: Before the LLM generates code, the system automatically runs `chmod -x` on critical files (e.g., `evaluator.py`, `strategy.py`) listed in the `no_exec_files` configuration of the node.
*   **Prompt Injection**: The system prompt is automatically updated to warn the AI: *"Do NOT try to run the code yourself."*
*   **Enforcement**: Even if the AI ignores the prompt and tries to run `./strategy.py`, the Operating System will deny permission.

## 2. Filesystem Snapshot & Rollback

Before any AI node executes, AgentCommander takes a "snapshot" of the current directory.

*   **Strict Mode**: Any file modification not explicitly allowed is reverted.
*   **Whitelist/Blacklist**: You can define granular rules (e.g., "Allow editing `strategy.py`, but forbid editing `evaluator.py`").
*   **Smart Ignoring**: The system intelligently ignores noise files like `__pycache__` to prevent false positive security warnings.

## 3. Parent Directory Lock

When `lock_parent` is enabled, the system temporarily removes write permissions (`chmod u-w`) from the parent directory of the experiment. This prevents the agent from "escaping" its sandbox and modifying sibling experiments or system files.

## 4. CLI Permissions (YOLO Mode)

By default, the underlying CLI tools operate in **YOLO mode** (e.g., using `yolo` or `skip-dangerous-permission` flags). This grants the model permission to use available tools autonomously within the working directory.

### Risks
*   **Arbitrary Privilege**: While AgentCommander restricts *filesystem* access via its sandbox, the agent still inherits the permissions of the OS user running the process.
*   **Prompt Injection**: Malicious or poorly designed prompts could theoretically guide an agent to perform unintended actions.

## Disclaimer

This software is provided "as is", without warranty of any kind. The developers are not responsible for any damage, loss of data, or security breaches. 

**Recommendation**: Always run AgentCommander inside a **Docker Container**, Virtual Machine, or a dedicated restricted user account to ensure complete isolation.