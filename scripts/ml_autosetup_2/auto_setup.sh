#!/bin/bash

# <GEMINI_UI_CONFIG>
# {
#   "name": "[Scenario 2] Bring Your Own Training Code",
#   "description": "Adapt your existing training script. The Agent will analyze your code to find data sources and optimize it based on your instructions.",
#   "inputs": [
#     {"id": "TARGET_ROOT_DIR", "label": "Target Root Directory (Where to create project)", "type": "text", "default": ".", "tooltip": "'.' = AgentCommander Root."},
#     {"id": "PROJECT_NAME", "label": "Project Name (e.g., my_new_experiment)", "type": "text", "default": "my_new_experiment", "tooltip": "Folder name to create"},
#     {"id": "TASK_TYPE", "label": "Task Type", "type": "select", "options": ["Standard (Regression/Classification)", "Reinforcement Learning"], "default": "Standard (Regression/Classification)"},
#     {"id": "REFERENCE_STRATEGY_FILE_PATH", "label": "Original Strategy File Path (.py)", "type": "text", "default": "your_code/strategy.py", "tooltip": "Absolute path to your existing .py training script."},
#     {"id": "STRATEGY_DIR", "label": "Strategy Dependency Folder (Optional)", "type": "text", "default": "", "tooltip": "Folder containing helper scripts or environment files. Agent can modify files here."},
#     {"id": "DATA_PROTOCOL_DESC", "label": "Data & Splitting Instructions (Natural Language)", "type": "textarea", "default": "Find the data source in the strategy file. Use 80% for training and 20% for testing.", "rows": 3},
#     {"id": "VENV_PYTHON", "label": "Python Interpreter Path (For splitting & config)", "type": "text", "default": "/home/liumx/.conda/envs/agent_commander/bin/python"},
#     {"id": "EVAL_CMD", "label": "Evaluation Command", "type": "text", "default": "/home/liumx/.conda/envs/agent_commander/bin/python strategy.py && /home/liumx/.conda/envs/agent_commander/bin/python evaluator.py", "tooltip": "Command to run training then evaluation. Sequential execution is required."},
#     {"id": "LLM_MODEL", "label": "LLM Model (for generation)", "type": "llm_selector", "options": ["__STANDARD_MODELS__"], "default": "auto-gemini-3"},
#     {"id": "LLM_TIMEOUT", "label": "LLM Generation Time Limit (s)", "type": "number", "default": 300, "tooltip": "Max time allowed for AI to generate code. Default is 300s."},
#     {"id": "LOCK_PARENT", "label": "🔒 Lock Parent Directory (Read-Only during generation)", "type": "radio", "options": ["true", "false"], "default": "false"},
#     {"id": "SOFT_LIMIT", "label": "Soft Time Limit (s) [Per Eval: No new searches start after this, but current trial finishes]", "type": "number", "default": 600},
#     {"id": "HARD_LIMIT", "label": "Hard Time Limit (s) [Per Eval: Kill immediately if exceeded, mark as Failure]", "type": "number", "default": 900},
#     {"id": "USER_SEED", "label": "Random Seed (Number or 'random')", "type": "text", "default": "42", "tooltip": "Enter a number or 'random'"},
#     {"id": "METRIC_TEXT", "label": "Metric Description (Defines calculate_score. System auto-converts to 'Lower is Better' e.g. via negative sign)", "type": "textarea", "default": "MSE", "rows": 2},
#     {"id": "TASK_BG_TEXT", "label": "Task Background (Optional, e.g. LSTM/CNN for 3D/4D data)", "type": "textarea", "default": "GW PTA wave to phase", "rows": 2},
#     {"id": "MODEL_HINT_TEXT", "label": "Model/Strategy Hint (Optional)", "type": "textarea", "default": "with cnn+LSTM?", "rows": 2}
#   ],
#   "preview_steps": [
#     "1. Environment Check & Confirmation",
#     "2. Create Directory Structure",
#     "3. AI Adaptation (Strategy, Evaluator, Metric, Plot).",
#     "4. Validation (Sequential Train -> Eval)",
#     "5. Update config.json"
#   ],
#   "system_intro": [
#     "PROTOCOL & ARCHITECTURE:",
#     "• experiment_setup.py: IMMUTABLE data protocol. Ensures Strategy and Evaluator use identical splits.",
#     "• strategy.py (PLAYER): Your training code. Must save weights and implement load_trained_model() for Evaluator.",
#     "• evaluator.py (JUDGE): Loads weights from Strategy and runs standardized evaluation. Includes Anti-Cheating check.",
#     "• metric.py & plot.py: Automated score and visualization logic. Read-Only during iteration loop."
#   ]
# }
# </GEMINI_UI_CONFIG>

# 1. Locate Source Files (Assumes script is in the same dir as templates)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EVALUATOR_SCRIPT="$SCRIPT_DIR/evaluator.py"
STRATEGY_SCRIPT="$SCRIPT_DIR/strategy.py"
METRIC_SCRIPT="$SCRIPT_DIR/metric.py"
PLOT_SCRIPT="$SCRIPT_DIR/plot.py"

# Reference Files
STRATEGY_REF="$SCRIPT_DIR/strategy_ref.py"
EXP_SETUP_SCRIPT="$SCRIPT_DIR/experiment_setup.py"

echo "=== ML Project Auto-Setup Wizard ==="
echo "Date: $(date)"
echo "------------------------------------"

# ==============================================================================
# 0. Initial Warning & Confirmation (Skipped if NON_INTERACTIVE is set)
# ==============================================================================
if [ -z "$NON_INTERACTIVE" ]; then
    echo "⚠️  WARNING: This script will OVERWRITE/UPDATE 'config.json' in the current directory."
    echo "------------------------------------"
    echo -e "\n[IMPORTANT] Workflow Evaluation Logic"
    echo "By default, this workflow executes evaluation in the 'Experiment Subloop'"
    echo "at node: '4. Run Evaluator' (ID: step4_eval)."
    echo ""
    echo "Default Command:"
    echo "--------------------------------------------------------------------------------"
    echo "cd {current_exp_path} && {eval_cmd}"
    echo "--------------------------------------------------------------------------------"
    echo ""
    echo "NOTE FOR SERVER/HPC USERS (e.g., QSUB, SLURM):"
    echo "If you need to submit jobs to compute nodes, you should modify the command in the"
    echo "Workflow Editor (step4_eval) to use a wrapper script that:"
    echo "  1. Submits the job (e.g., qsub run_job.sh)"
    echo "  2. WAITS for the job to complete (polling until done)"
    echo "  3. Prints the final output so the agent can parse 'Best metric: X.XXX'"
    echo ""
    read -p "Press [Enter] to confirm you understand this and continue setup..." dummy_var
    echo ""
else
    echo "ℹ️  Running in Non-Interactive Mode (UI Automation)"
fi

# ==============================================================================
# 2. User Inputs (Environment Variable Priority)
# ==============================================================================

# Helper function to get input if variable is not set
get_input() {
    local var_name=$1
    local prompt=$2
    local default=$3
    local current_val=${!var_name}

    if [ -z "$current_val" ]; then
        if [ -n "$NON_INTERACTIVE" ]; then
             if [ -n "$default" ]; then
                 eval "$var_name=\"$default\""
                 echo "$prompt: $default (Default used in Non-Interactive mode)"
             else
                 echo "❌ Error: Required field '$var_name' is missing in Non-Interactive mode."
                 exit 1
             fi
        elif [ -n "$default" ]; then
             read -p "$prompt [Default: $default]: " user_val
             if [ -z "$user_val" ]; then
                 eval "$var_name=\"$default\""
             else
                 eval "$var_name=\"$user_val\""
             fi
        else
             while true; do
                 read -p "$prompt: " user_val
                 if [ -n "$user_val" ]; then
                     eval "$var_name=\"$user_val\""
                     break
                 fi
                 echo "Error: This field is mandatory."
             done
        fi
    else
        echo "$prompt: $current_val (Loaded from Env)"
    fi
}

# ==============================================================================
# 1.5. Navigate to Target Root
# ==============================================================================
get_input "TARGET_ROOT_DIR" "Target Root Directory" "."

if [ -n "$TARGET_ROOT_DIR" ]; then
    if [ "$TARGET_ROOT_DIR" == "." ]; then
        if [ -n "$AGENT_APP_ROOT" ]; then
            echo "[Setup] Switching to AgentCommander Root: $AGENT_APP_ROOT"
            cd "$AGENT_APP_ROOT" || exit 1
        else
            echo "[Setup] Staying in current directory (CLI mode)"
        fi
    else
        # Expand tilde if present
        if [[ "$TARGET_ROOT_DIR" == "~"* ]]; then TARGET_ROOT_DIR="${TARGET_ROOT_DIR/#\~/$HOME}"; fi
        
        echo "[Setup] Switching to Target Root: $TARGET_ROOT_DIR"
        mkdir -p "$TARGET_ROOT_DIR"
        cd "$TARGET_ROOT_DIR" || exit 1
    fi
fi

get_input "PROJECT_NAME" "[REQUIRED] Project Name (e.g., my_new_experiment)" ""
get_input "TASK_TYPE" "Task Type" "Standard (Regression/Classification)"
get_input "LLM_TIMEOUT" "LLM Generation Time Limit (s)" "300"
get_input "REFERENCE_STRATEGY_FILE_PATH" "Original Strategy File Path (.py)" ""
get_input "STRATEGY_DIR" "Strategy Dependency Folder" ""
get_input "DATA_PROTOCOL_DESC" "Data Handling Instructions" ""

# Set Reference Templates based on Task Type
if [[ "$TASK_TYPE" == *"Reinforcement"* ]]; then
    EVALUATOR_REF="$SCRIPT_DIR/evaluator_rl_ref.py"
    METRIC_REF="$SCRIPT_DIR/metric_rl_ref.py"
    PLOT_REF="$SCRIPT_DIR/plot_rl_ref.py"
    echo "Task Type: Reinforcement Learning detected. Using RL Templates."
else
    EVALUATOR_REF="$SCRIPT_DIR/evaluator_std_ref.py"
    METRIC_REF="$SCRIPT_DIR/metric_std_ref.py"
    PLOT_REF="$SCRIPT_DIR/plot_std_ref.py"
    echo "Task Type: Standard ML detected. Using Standard Templates."
fi

# Check if source files exist
for file in "$EVALUATOR_REF" "$METRIC_REF" "$PLOT_REF" "$STRATEGY_SCRIPT" "$STRATEGY_REF" "$EXP_SETUP_SCRIPT"; do
    if [ ! -f "$file" ]; then
        echo "Error: Source file not found: $file"
        exit 1
    fi
done

# Validation for Strategy Dependency Folder
if [ -n "$STRATEGY_DIR" ]; then
    if [[ "$STRATEGY_DIR" == "~"* ]]; then STRATEGY_DIR="${STRATEGY_DIR/#\~/$HOME}"; fi
    if [ ! -d "$STRATEGY_DIR" ]; then
        echo "❌ Error: Strategy dependency folder not found: $STRATEGY_DIR"
        exit 1
    fi
    echo "Using strategy dependencies from: $STRATEGY_DIR"
fi

# Validation for Original Strategy File
if [ -n "$REFERENCE_STRATEGY_FILE_PATH" ]; then
    if [[ "$REFERENCE_STRATEGY_FILE_PATH" == "~"* ]]; then REFERENCE_STRATEGY_FILE_PATH="${REFERENCE_STRATEGY_FILE_PATH/#\~/$HOME}"; fi
    if [ ! -f "$REFERENCE_STRATEGY_FILE_PATH" ]; then
        echo "❌ Error: Original strategy file not found: $REFERENCE_STRATEGY_FILE_PATH"
        exit 1
    fi
    echo "Using source code from: $REFERENCE_STRATEGY_FILE_PATH"
    # Overwrite the template's strategy_ref with user's code
    cp "$REFERENCE_STRATEGY_FILE_PATH" "$STRATEGY_REF"
fi

DEFAULT_VENV="/home/$USER/.conda/envs/agent_commander/bin/python"
get_input "VENV_PYTHON" "Python Interpreter Path" "$DEFAULT_VENV"

# ... (omitted evaluation config)

# ==============================================================================
# 3. Directory Structure & File Copying
# ==============================================================================

echo -e "\n[Setup] Creating directories..."
PROJECT_ROOT="./$PROJECT_NAME"

# Check if project directory already exists
if [ -d "$PROJECT_ROOT" ]; then
    FULL_PATH=$(realpath "$PROJECT_ROOT")
    echo "❌ Error: Project directory '$PROJECT_ROOT' already exists at: $FULL_PATH"
    echo "Please choose a different project name or delete the existing directory."
    exit 1
fi

EXP_DIR="$PROJECT_ROOT/Branch_example/exp_example"

mkdir -p "$EXP_DIR/data" 
if [ -n "$STRATEGY_DIR" ]; then
    echo "[Setup] Copying strategy dependencies to $EXP_DIR/strategy_lib..."
    mkdir -p "$EXP_DIR/strategy_lib"
    cp -r "$STRATEGY_DIR"/* "$EXP_DIR/strategy_lib/"
fi

cp "$STRATEGY_SCRIPT" "$EXP_DIR/"
# Copy dynamically selected Evaluator, Metric, and Plot
cp "$EVALUATOR_REF" "$EXP_DIR/evaluator.py"
cp "$METRIC_REF" "$EXP_DIR/metric.py"
cp "$PLOT_REF" "$EXP_DIR/plot.py"
# Copy Reference & Setup files
cp "$STRATEGY_REF" "$EVALUATOR_REF" "$EXP_SETUP_SCRIPT" "$EXP_DIR/"

echo "[Setup] Files copied to $EXP_DIR"

# ==============================================================================
# 4. Data Protocol Generation (AI Driven)
# ==============================================================================

echo -e "\n[Step 4] AI Analyzing data sources and generating protocol..."

printf "Task: Analyze '$EXP_DIR/strategy_ref.py' to find where it loads data from (e.g., file paths, numpy arrays). \
Now, implement 'load_and_split_data()' in '$EXP_DIR/experiment_setup.py'. \
Instruction from user: $DATA_PROTOCOL_DESC. \
Contract: The function must return (X_train, X_test, y_train, y_test) or appropriate RL equivalents. \
IMPORTANT: Hardcode any discovered absolute paths into the generated code to ensure portability." | python3 "$AGENT_APP_ROOT/scripts/llm_runner.py" \
    --model "$LLM_MODEL" \
    --cwd "$EXP_DIR" \
    --whitelist "strategy.py,metric.py,plot.py,strategy_lib/,experiment_setup.py" \
    --timeout "$LLM_TIMEOUT"

# ==============================================================================
# 5. Configure Evaluator (Random Seed only)
# ==============================================================================

TARGET_SETUP="$EXP_DIR/experiment_setup.py"
# Update Random Seed
sed -i "s/^PROTOCOL_SEED = .*/PROTOCOL_SEED = $RANDOM_SEED/" "$TARGET_SETUP"

echo "Experiment Setup initialized."

# ==============================================================================
# 6. AI Generation Loop
# ==============================================================================

RETRY_COUNT=0
LAST_ERROR_LOG=""

while true; do
    echo -e "\n========================================"
    echo "   Starting AI Code Generation..."
    echo "========================================"

    # --- Lock Logic ---
    LOCK_FLAG=""
    if [ "$LOCK_PARENT" == "true" ]; then LOCK_FLAG="--lock-parent"; fi
    
    # Restrict execution of key files during generation
    NO_EXEC_FLAG="--no-exec evaluator.py,strategy.py"

    # --- Reset Core Files (Task-Specific) ---
    cp "$STRATEGY_REF" "$EXP_DIR/strategy.py"
    cp "$EVALUATOR_REF" "$EXP_DIR/evaluator.py"
    cp "$METRIC_REF" "$EXP_DIR/metric.py"
    cp "$PLOT_REF" "$EXP_DIR/plot.py"
    
    # Robust Variable Injection for experiment_setup.py
    TARGET_SETUP="$EXP_DIR/experiment_setup.py"
    FINAL_SEED=${RANDOM_SEED:-42}
    if grep -q "PROTOCOL_SEED =" "$TARGET_SETUP"; then
        sed -i "s/^PROTOCOL_SEED = .*/PROTOCOL_SEED = $FINAL_SEED/" "$TARGET_SETUP"
    fi
    
    # --- Step 5: Strategy Generation ---
    echo "[LLM] Generating Strategy (Attempt $((RETRY_COUNT+1)))..."
    
    EXTRA_INSTRUCTION=""
    RESUME_FLAG=""
    
    if [ $RETRY_COUNT -gt 0 ]; then
        RESUME_FLAG="--resume"
        if [ -n "$LAST_ERROR_LOG" ]; then
            EXTRA_INSTRUCTION="PREVIOUS ATTEMPT FAILED. Error Log:\n$LAST_ERROR_LOG\n\nFix the code based on this error."
            echo -e "\n" + "#"*40 + " [DEBUG: FEEDBACK TO AI] " + "#"*40
            echo -e "$EXTRA_INSTRUCTION"
            echo -e "#"*100 + "\n"
        fi
    fi

    # Tailor Prompt based on Task Type
    if [[ "$TASK_TYPE" == *"Reinforcement"* ]]; then
        RL_DATA_RESTRICTION="**SPECIAL RL RULE**: The data loading logic in 'strategy_lib/env.py' is already correctly configured with absolute paths. Do NOT attempt to replace it with 'experiment_setup.py' logic. Focus only on implementing the mandatory strategy interfaces."
    else
        RL_DATA_RESTRICTION="1. DATA: Replace original data loading with 'from experiment_setup import load_and_split_data' to align with the framework protocol."
    fi

    PROMPT_STRATEGY="Target: $EXP_DIR/strategy.py. Task Background: $TASK_BG_TEXT. Model Hints: $MODEL_HINT_TEXT. \
GOAL: Adapt this user-provided script to our Agent Framework with **EXTREMELY MINIMAL** changes. Do NOT refactor the core logic unless absolutely necessary for the script to run. \
REQUIREMENTS: \
$RL_DATA_RESTRICTION \
2. INTERFACE: Implement 'def load_trained_model(path, device):' (refer to strategy_ref.py for the signature). This is ONLY for model weight loading. \
3. EXECUTION: Ensure the training process saves the model to 'best_fast.pt' (or appropriate format) and can be triggered by 'if __name__ == \"__main__\":'. \
4. CONTEXT: If a 'strategy_lib' directory exists, it contains helper files. You may make minor adjustments there ONLY if they are required to support the interfaces above. \
5. CRITICAL: Preserve the original model architecture and training hyperparameters as much as possible. Adaptation is the priority, not optimization at this stage. \
6. MANDATORY: The functions 'load_trained_model' and the 'if __name__ == \"__main__\":' training block are **MANDATORY**. Do NOT remove them to fix run errors. If you encounter errors, fix the underlying environment or data logic instead. \
7. IMPORTANT: Do NOT try to run the code yourself. The system will run it for you after you finish editing. \
$EXTRA_INSTRUCTION"
    
    # Define consistent whitelist for all steps
    GLOBAL_WHITELIST="strategy.py,metric.py,plot.py,strategy_lib/,experiment_setup.py"

    printf "%b" "$PROMPT_STRATEGY" | python3 "$AGENT_APP_ROOT/scripts/llm_runner.py" \
        --model "$LLM_MODEL" \
        --cwd "$EXP_DIR" \
        --whitelist "$GLOBAL_WHITELIST" \
        --timeout "$LLM_TIMEOUT" \
        $LOCK_FLAG \
        $NO_EXEC_FLAG \
        $RESUME_FLAG

    # --- Step 5.5: Evaluator Adaptation (DEPRECATED: We use standardized engine) ---
    echo "[Info] Using standardized Evaluator Engine. No adaptation needed."

    # --- Step 6: Metric Generation ---
    echo "[LLM] Generating Metric logic..."
    if [[ "$TASK_TYPE" == *"Reinforcement"* ]]; then
        PROMPT_METRIC="Goal: Implement 'calculate_rl_score(history)' in $EXP_DIR/metric.py. \
        Hint: $METRIC_TEXT. History is a list of dicts from env.step info. Return a scalar float (higher is usually better)."
    else
        PROMPT_METRIC="Goal: Implement 'calculate_standard_score(y_true, y_pred)' in $EXP_DIR/metric.py. \
        Hint: $METRIC_TEXT. Use numpy. Return a scalar float (MSE, Accuracy, etc.)."
    fi
    
    printf "%b" "$PROMPT_METRIC" | python3 "$AGENT_APP_ROOT/scripts/llm_runner.py" \
        --model "$LLM_MODEL" \
        --cwd "$EXP_DIR" \
        --whitelist "$GLOBAL_WHITELIST" \
        --timeout "$LLM_TIMEOUT" \
        $LOCK_FLAG \
        $NO_EXEC_FLAG \
        --resume
    
    # --- Step 7: Plot Generation ---
    echo "[LLM] Generating Plot visualization..."
    if [[ "$TASK_TYPE" == *"Reinforcement"* ]]; then
        PROMPT_PLOT="Goal: Implement 'draw_rl_plots(history, output_dir)' in $EXP_DIR/plot.py. \
        Task Background: $TASK_BG_TEXT. Draw a professional plot (e.g., Equity Curve) and save as 'best_result.png'."
    else
        PROMPT_PLOT="Goal: Implement 'draw_standard_plots(X, y_true, y_pred, output_dir)' in $EXP_DIR/plot.py. \
        Task Background: $TASK_BG_TEXT. Draw a professional plot (e.g., Pred vs True) and save as 'best_result.png'."
    fi
    
    printf "%b" "$PROMPT_PLOT" | python3 "$AGENT_APP_ROOT/scripts/llm_runner.py" \
        --model "$LLM_MODEL" \
        --cwd "$EXP_DIR" \
        --whitelist "$GLOBAL_WHITELIST" \
        --timeout "$LLM_TIMEOUT" \
        $LOCK_FLAG \
        $NO_EXEC_FLAG \
        --resume

    # ==============================================================================
    # 7-9. Validation & Integrity Checks
    # ==============================================================================
    
    echo -e "\n[Validation] Checking integrity and functionality..."
    HAS_ERROR=0

    # 7. Check Strategy Interface
    if ! grep -q "def load_trained_model" "$EXP_DIR/strategy.py"; then
        echo "⚠️  WARNING: strategy.py is missing 'def load_trained_model'. Triggering surgical fix..."
        PROMPT_FIX="The file $EXP_DIR/strategy.py is missing the MANDATORY 'def load_trained_model(path, device)' function. \
        Please ADD this function to the end of the file. It should load the model weights from the given path and return the model instance. \
        Refer to the Task Background and original strategy for the model class name."
        
        printf "%b" "$PROMPT_FIX" | python3 "$AGENT_APP_ROOT/scripts/llm_runner.py" \
            --model "$LLM_MODEL" \
            --cwd "$EXP_DIR" \
            --whitelist "strategy.py" \
            --timeout "$LLM_TIMEOUT" \
            $LOCK_FLAG \
            $NO_EXEC_FLAG \
            --resume
            
        # Re-check
        if ! grep -q "def load_trained_model" "$EXP_DIR/strategy.py"; then
            echo "❌ ERROR: Surgical fix failed to restore 'load_trained_model'."
            HAS_ERROR=1
        fi
    fi

    # 8. Check Evaluator Engine (Task-Aware)
    if [[ "$TASK_TYPE" == *"Reinforcement"* ]]; then
        # RL Check: Verify the engine calls the metric interface
        if ! grep -q "calculate_rl_score" "$EXP_DIR/evaluator.py"; then
            echo "❌ ERROR: evaluator.py lost the RL metric interface!"
            HAS_ERROR=1
        fi
    else
        # Standard Check: Anti-Cheating Protection
        if ! grep -q "def check_data_leakage" "$EXP_DIR/evaluator.py"; then
            echo "❌ ERROR: evaluator.py is missing 'def check_data_leakage'!"
            HAS_ERROR=1
        fi
        if ! grep -q "check_data_leakage(" "$EXP_DIR/evaluator.py"; then
            echo "❌ ERROR: evaluator.py defines but NEVER CALLS 'check_data_leakage'!"
            HAS_ERROR=1
        fi
    fi

    # 9. Dry Run (Sequential)
    echo "[Validation] performing dry run (Train -> Eval)..."
    
    # Use subshell to isolate directory change and prevent path corruption
    (
        cd "$EXP_DIR" || exit 1
        
        # 1. Run Training
        echo "Running Strategy (Training)..."
        if ! "$VENV_PYTHON" strategy.py > train_log.txt 2>&1; then
            echo "❌ Training Failed."
            cat train_log.txt | tail -n 20
            exit 101
        else
            echo "✅ Training Complete."
        fi
        
        # 2. Run Evaluation
        echo "Running Evaluator..."
        if ! "$VENV_PYTHON" evaluator.py > eval_out.txt 2>&1; then
            echo "❌ Evaluation Failed."
            cat eval_out.txt
            exit 102
        else
            echo "✅ Evaluation Complete."
            OUT=$(cat eval_out.txt)
            echo "$OUT"
            
            # Check for metric output
            if ! echo "$OUT" | grep -q "Best metric:"; then
                 echo "❌ Critical: Evaluator did not print 'Best metric: X.XXX'"
                 exit 103
            fi
            
            if echo "$OUT" | grep -q "Best metric: inf"; then
                 echo "❌ Critical: Metric is INF."
                 exit 104
            fi
        fi
    )
    
    # Capture subshell exit code
    DRY_RUN_EXIT_CODE=$?
    
    # Debug info
    if [ $DRY_RUN_EXIT_CODE -ne 0 ]; then
        echo "DEBUG: Subshell exited with code $DRY_RUN_EXIT_CODE."
        if [ $DRY_RUN_EXIT_CODE -eq 101 ]; then echo "DEBUG: Failure Point -> Strategy Training"; fi
        if [ $DRY_RUN_EXIT_CODE -eq 102 ]; then echo "DEBUG: Failure Point -> Evaluator Execution"; fi
        if [ $DRY_RUN_EXIT_CODE -eq 103 ]; then echo "DEBUG: Failure Point -> Missing 'Best metric' output"; fi
        if [ $DRY_RUN_EXIT_CODE -eq 104 ]; then echo "DEBUG: Failure Point -> Metric is INF"; fi
    fi
    
    if [ $DRY_RUN_EXIT_CODE -eq 0 ]; then
        echo "✅ Dry Run Successful."
    else
        echo "❌ Dry Run Failed (Code $DRY_RUN_EXIT_CODE)."
        HAS_ERROR=1
        # Capture errors for next attempt feedback
        if [ -f "$EXP_DIR/train_log.txt" ]; then
            LAST_ERROR_LOG=$(tail -n 50 "$EXP_DIR/train_log.txt" "$EXP_DIR/eval_out.txt" 2>/dev/null)
        fi
    fi

    # ==============================================================================
    # 10. Result & Retry Prompt
    # ==============================================================================

    if [ $HAS_ERROR -eq 1 ]; then
        if [ $RETRY_COUNT -lt 5 ]; then
            RETRY_COUNT=$((RETRY_COUNT+1))
            echo "⚠️  Errors detected. Auto-retrying ($RETRY_COUNT/5) with error feedback..."
            
            # Capture error log for feedback
            LAST_ERROR_LOG=$(echo "$DRY_RUN_OUT" | tail -n 50)
            
            sleep 2
            continue
        else
            echo -e "\n❌ Auto-retry failed 5 times."
            echo "----------------------------------------------------------------------"
            echo "[SUGGESTION]"
            echo "1. Review the error logs above carefully for code or environment issues."
            echo "2. Verify your Data (X.npy, Y.npy) is not corrupted and contains valid values."
            echo "3. Check if your Python Interpreter Path and dependencies are correct."
            echo "4. You can try to run the Setup again with more specific Model Hints."
            echo "----------------------------------------------------------------------"
            echo "Exiting with errors."
            break
        fi
    else
        echo -e "\n🎉 Setup and initial verification complete! No obvious bugs found."
        echo "Work directory: $EXP_DIR"
        break
    fi

done

# ==============================================================================
# 11. Auto-Configure config.json
# ==============================================================================

echo -e "\n[Config] Auto-updating config.json..."

# Get Plot Filename from Evaluator (Dry Run)
PLOT_OUTPUT=""
if [ -f "$EXP_DIR/evaluator.py" ]; then
    # Run in the exp dir to ensure relative imports work
    current_dir=$(pwd)
    cd "$EXP_DIR" || exit
    # Capture output and trim whitespace
    # We remove 2>/dev/null to see errors if this fails
    PLOT_OUTPUT=$("$VENV_PYTHON" "evaluator.py" --dry-run-plot | tr -d '[:space:]')
    echo "[Debug] Detected Plot Output: $PLOT_OUTPUT"
    cd "$current_dir" || exit
fi

# Export vars for Python script
export PROJECT_NAME
export VENV_PYTHON
export EVAL_CMD
export PLOT_OUTPUT
export TASK_BG_TEXT
export METRIC_TEXT
export AGENT_APP_ROOT

python3 - <<'EOF'
import json
import os
from pathlib import Path

# Use the environment variable to find the absolute app root
app_root = os.environ.get('AGENT_APP_ROOT', os.getcwd())
config_path = os.path.join(app_root, 'config.json')
template_path = os.path.join(app_root, 'config_template.json')

try:
    data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            try: data = json.load(f)
            except: data = {}
    elif os.path.exists(template_path):
        with open(template_path, 'r') as f:
            try: data = json.load(f)
            except: data = {}
    
    if 'global_vars' not in data: data['global_vars'] = {}

    # 1. Update root_dir with ABSOLUTE PATH to ensure the Agent can always find it
    # We are currently in the Target Root (where PROJECT_NAME folder was created)
    project_abs_path = os.path.abspath(os.path.join(os.getcwd(), os.environ['PROJECT_NAME']))
    data['root_dir'] = project_abs_path
    
    # 2. Update Global Variables
    data['global_vars']['venv'] = os.environ['VENV_PYTHON']
    
    # Use the custom EVAL_CMD from UI, or fall back to the sequential template
    py = os.environ['VENV_PYTHON']
    default_cmd = f"{py} strategy.py && {py} evaluator.py"
    data['global_vars']['eval_cmd'] = os.environ.get('EVAL_CMD', default_cmd)
    
    # 3. Handle Plot Names
    plot_out = os.environ.get('PLOT_OUTPUT', '')
    if not plot_out:
        plot_out = "@best_result.png"
    data['global_vars']['plot_names'] = plot_out
    
    # 4. Construct System Prompt
    task = os.environ.get('TASK_BG_TEXT', '')
    metric = os.environ.get('METRIC_TEXT', '')
    sys_instruction = (
        "1. You can improve by modifying Model Architecture, Hyperparameter Search, and Reward/Feature logic.\n"
        "2. Add debug info and SAVE worst samples/predictions as .npy files for analysis.\n"
        "3. Optimize for speed; avoid redundancy."
    )
    data['global_vars']['DEFAULT_SYS'] = f"You are an expert AI Data Scientist. Task: {task}. Metric: {metric}. Goal: Optimize strategy.py. \n{sys_instruction}"

    # Write back to the main config.json
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ config.json updated successfully at: {config_path}")
    print(f"📍 New Project Root: {data['root_dir']}")
except Exception as e:
    print(f"❌ Failed to update config.json: {e}")
EOF

echo "Done."