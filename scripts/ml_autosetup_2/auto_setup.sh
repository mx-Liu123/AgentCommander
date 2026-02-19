#!/bin/bash

# <GEMINI_UI_CONFIG>
# {
#   "name": "[Case: You have Training Code] ML Auto-Setup",
#   "description": "Adapt your existing ML training script to the Agent framework. Requires your script to save model weights (e.g. .pt) for independent evaluation.",
#   "inputs": [
#     {"id": "TARGET_ROOT_DIR", "label": "Target Root Directory (Where to create project)", "type": "text", "default": ".", "tooltip": "'.' = AgentCommander Root. Use absolute path for others."},
#     {"id": "PROJECT_NAME", "label": "Project Name (e.g., my_new_experiment)", "type": "text", "default": "my_new_experiment", "tooltip": "Folder name to create"},
#     {"id": "DATA_DIR", "label": "Data Directory (Absolute path, must contain X.npy and Y.npy)", "type": "text", "default": "~/test_data/", "tooltip": "Absolute path containing X.npy and Y.npy"},
#     {"id": "VENV_PYTHON", "label": "Python Interpreter Path (For splitting & config)", "type": "text", "default": "/home/liumx/.conda/envs/agent_commander/bin/python"},
#     {"id": "EVAL_CMD", "label": "Evaluation Command", "type": "text", "default": "/home/liumx/.conda/envs/agent_commander/bin/python strategy.py && /home/liumx/.conda/envs/agent_commander/bin/python evaluator.py", "tooltip": "Command to run training then evaluation. Sequential execution is required."},
#     {"id": "LLM_MODEL", "label": "LLM Model (for generation)", "type": "llm_selector", "options": ["__STANDARD_MODELS__"], "default": "auto-gemini-3"},
    {"id": "LOCK_PARENT", "label": "🔒 Lock Parent Directory (Read-Only during generation)", "type": "radio", "options": ["true", "false"], "default": "false"},
    {"id": "SOFT_LIMIT", "label": "Soft Time Limit (s) [Per Eval: No new searches start after this, but current trial finishes]", "type": "number", "default": 600},
    {"id": "HARD_LIMIT", "label": "Hard Time Limit (s) [Per Eval: Kill immediately if exceeded, mark as Failure]", "type": "number", "default": 900},
    {"id": "USER_SEED", "label": "Random Seed (Number or 'random')", "type": "text", "default": "42", "tooltip": "Enter a number or 'random'"},
    {"id": "METRIC_TEXT", "label": "Metric Description (Defines calculate_score. System auto-converts to 'Lower is Better' e.g. via negative sign)", "type": "textarea", "default": "MSE", "rows": 2},
    {"id": "TASK_BG_TEXT", "label": "Task Background (Optional, e.g. LSTM/CNN for 3D/4D data)", "type": "textarea", "default": "GW PTA wave to phase", "rows": 2},
    {"id": "MODEL_HINT_TEXT", "label": "Model/Strategy Hint (Optional)", "type": "textarea", "default": "with cnn+LSTM?", "rows": 2}
  ],
  "preview_steps": [
    "1. Environment Check & Confirmation",
    "2. Create Directory Structure",
    "3. AI Adaptation (Strategy, Evaluator, Metric, Plot).",
    "4. Validation (Sequential Train -> Eval)",
    "5. Update config.json"
  ],
  "system_intro": [
    "PROTOCOL & ARCHITECTURE:",
    "• experiment_setup.py: IMMUTABLE data protocol. Ensures Strategy and Evaluator use identical splits.",
    "• strategy.py (PLAYER): Your training code. Must save weights and implement load_trained_model() for Evaluator.",
    "• evaluator.py (JUDGE): Loads weights from Strategy and runs standardized evaluation. Includes Anti-Cheating check.",
    "• metric.py & plot.py: Automated score and visualization logic. Read-Only during iteration loop."
  ]
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
EVALUATOR_REF="$SCRIPT_DIR/evaluator_ref.py"
EXP_SETUP_SCRIPT="$SCRIPT_DIR/experiment_setup.py"

# Check if source files exist
for file in "$EVALUATOR_SCRIPT" "$STRATEGY_SCRIPT" "$METRIC_SCRIPT" "$PLOT_SCRIPT" "$STRATEGY_REF" "$EVALUATOR_REF" "$EXP_SETUP_SCRIPT"; do
    if [ ! -f "$file" ]; then
        echo "Error: Source file not found: $file"
        exit 1
    fi
done

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

# Special handling for DATA_DIR validation
if [ -z "$DATA_DIR" ]; then
    while true; do
        echo -e "\n[REQUIRED] Enter the ABSOLUTE path to the Data Directory."
        read -p "Data Directory (must contain X.npy and Y.npy): " DATA_DIR
        # Validation Logic...
        if [[ "$DATA_DIR" == "~"* ]]; then DATA_DIR="${DATA_DIR/#\~/$HOME}"; fi
        if [[ "$DATA_DIR" == ~* ]]; then DATA_DIR="${DATA_DIR/#\~/$HOME}"; fi
        DATA_DIR=$(realpath "$DATA_DIR" 2>/dev/null)
        
        if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
            echo "Error: Invalid path: $DATA_DIR"
            DATA_DIR="" # Reset to loop
            continue
        fi
        if [ ! -f "$DATA_DIR/X.npy" ] || [ ! -f "$DATA_DIR/Y.npy" ]; then
            echo "Error: Missing X.npy or Y.npy in $DATA_DIR"
            DATA_DIR=""
            continue
        fi
        break
    done
else
    # Env Var set, validate it once
    echo "Data Directory: $DATA_DIR (Loaded from Env)"
    if [[ "$DATA_DIR" == "~"* ]]; then DATA_DIR="${DATA_DIR/#\~/$HOME}"; fi
    DATA_DIR=$(realpath "$DATA_DIR" 2>/dev/null)
    if [ ! -d "$DATA_DIR" ] || [ ! -f "$DATA_DIR/X.npy" ]; then
        echo "❌ Error: Invalid DATA_DIR from environment: $DATA_DIR"
        exit 1
    fi
fi

DEFAULT_VENV="/home/$USER/.conda/envs/agent_commander/bin/python"
get_input "PYTHON_PATH" "Python Interpreter Path" "$DEFAULT_VENV"

echo -e "\n--- Evaluation Config ---"

DEFAULT_SOFT=600
get_input "SOFT_LIMIT" "Soft Time Limit (seconds)" "$DEFAULT_SOFT"

DEFAULT_HARD=900
get_input "HARD_LIMIT" "Hard Time Limit (seconds)" "$DEFAULT_HARD"

# Random Seed Logic
DEFAULT_SEED=42
if [ -z "$USER_SEED" ]; then
    read -p "Random Seed (Press Enter for $DEFAULT_SEED, or type 'random' for random): " USER_SEED
fi
if [ "$USER_SEED" == "random" ]; then
    RANDOM_SEED=$RANDOM
    echo "Using generated random seed: $RANDOM_SEED"
else
    RANDOM_SEED=${USER_SEED:-$DEFAULT_SEED}
    echo "Random Seed: $RANDOM_SEED"
fi

echo -e "\n--- AI Instructions ---"
get_input "METRIC_TEXT" "[REQUIRED] Metric Description" ""

echo -e "\nTip: Task Background..."
get_input "TASK_BG_TEXT" "Task Background (Optional)" ""
get_input "MODEL_HINT_TEXT" "Model/Strategy Hint (Optional)" ""

# ==============================================================================
# 3. Directory Structure & File Copying
# ==============================================================================

echo -e "\n[Setup] Creating directories..."
PROJECT_ROOT="./$PROJECT_NAME"

# Check if project directory already exists
if [ -d "$PROJECT_ROOT" ]; then
    echo "❌ Error: Project directory '$PROJECT_ROOT' already exists!"
    echo "Please chose a different project name or delete the existing directory."
    exit 1
fi

EXP_DIR="$PROJECT_ROOT/Branch_example/exp_example"

mkdir -p "$EXP_DIR/data" # Create data dir placeholder, though we use absolute path
cp "$EVALUATOR_SCRIPT" "$STRATEGY_SCRIPT" "$METRIC_SCRIPT" "$PLOT_SCRIPT" "$EXP_DIR/"
# Copy Reference & Setup files
cp "$STRATEGY_REF" "$EVALUATOR_REF" "$EXP_SETUP_SCRIPT" "$EXP_DIR/"

echo "[Setup] Files copied to $EXP_DIR"

# ==============================================================================
# 5. Configure Experiment Setup (sed)
# ==============================================================================

TARGET_SETUP="$EXP_DIR/experiment_setup.py"
echo "[Config] Injecting settings into $TARGET_SETUP..."

# 1. Update DATA_PATH
# We append a slash to ensure it's treated as a directory
sed -i "s|^DATA_PATH = .*|DATA_PATH = \"$DATA_DIR/\"|" "$TARGET_SETUP"

# 2. Update Random Seed
sed -i "s/^PROTOCOL_SEED = .*/PROTOCOL_SEED = $RANDOM_SEED/" "$TARGET_SETUP"

echo "Experiment Setup configured."

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

    # --- Reset to Reference Templates ---
    # We use the _ref files as the baseline for AI modification
    cp "$STRATEGY_REF" "$EXP_DIR/strategy.py"
    cp "$EVALUATOR_REF" "$EXP_DIR/evaluator.py"
    
    # --- Step 5: Strategy Generation ---
    echo "[LLM] Generating Strategy (Attempt $((RETRY_COUNT+1)))..."
    
    EXTRA_INSTRUCTION=""
    RESUME_FLAG=""
    
    if [ $RETRY_COUNT -gt 0 ]; then
        RESUME_FLAG="--resume"
        if [ -n "$LAST_ERROR_LOG" ]; then
            EXTRA_INSTRUCTION="PREVIOUS ATTEMPT FAILED. Error Log:\n$LAST_ERROR_LOG\n\nFix the code based on this error."
        fi
    fi

    PROMPT_STRATEGY="Target: $EXP_DIR/strategy.py. Task Background: $TASK_BG_TEXT. Model Hints: $MODEL_HINT_TEXT. \
GOAL: Adapt this user-provided script to our Agent Framework with MINIMAL changes. \
REQUIREMENTS: \
1. DATA: Replace original data loading with 'from experiment_setup import load_and_split_data'. \
2. INTERFACE: Implement 'def load_trained_model(path, device):' (See strategy_ref.py). This MUST return a loaded model instance for evaluation. \
3. EXECUTION: Ensure 'if __name__ == \"__main__\":' runs training and saves the model to 'best_fast.pt'. \
4. IMPORTANT: Do NOT try to run the code yourself. The system will run it for you after you finish editing. \
$EXTRA_INSTRUCTION"
    
    printf "%b" "$PROMPT_STRATEGY" | python3 "$AGENT_APP_ROOT/scripts/llm_runner.py" \
        --model "$LLM_MODEL" \
        --cwd "$EXP_DIR" \
        --whitelist "strategy.py,metric.py,plot.py" \
        --timeout 300 \
        $LOCK_FLAG \
        $NO_EXEC_FLAG \
        $RESUME_FLAG

    # --- Step 5.5: Evaluator Adaptation ---
    echo "[LLM] Adapting Evaluator..."
    PROMPT_EVAL="Target: $EXP_DIR/evaluator.py. Reference: $EXP_DIR/evaluator_ref.py. \
GOAL: Adapt the evaluator to test the model trained by strategy.py. \
REQUIREMENTS: \
1. Use 'from strategy import load_trained_model'. \
2. Use 'from experiment_setup import load_and_split_data, get_validation_noise_generator'. \
3. Load 'best_fast.pt' using the factory function. \
4. Perform inference and print 'Best metric: X.XXXX'. \
5. Do NOT try to run the code yourself."

    printf "%b" "$PROMPT_EVAL" | python3 "$AGENT_APP_ROOT/scripts/llm_runner.py" \
        --model "$LLM_MODEL" \
        --cwd "$EXP_DIR" \
        --whitelist "evaluator.py,metric.py,plot.py" \
        --timeout 300 \
        $LOCK_FLAG \
        $NO_EXEC_FLAG \
        --resume

    # --- Step 6: Metric Generation ---
    echo "[LLM] Generating Metric..."
    PROMPT_METRIC="Hint: $METRIC_TEXT. Now modify $EXP_DIR/metric.py. Ensure calculate_score(y_true, y_pred) handles the shapes produced by strategy.py."
    
    printf "%b" "$PROMPT_METRIC" | python3 "$AGENT_APP_ROOT/scripts/llm_runner.py" \
        --model "$LLM_MODEL" \
        --cwd "$EXP_DIR" \
        --whitelist "metric.py,plot.py" \
        --timeout 300 \
        $LOCK_FLAG \
        $NO_EXEC_FLAG \
        --resume
    
    # --- Step 7: Plot Generation (New) ---
    echo "[LLM] Generating Plot Visualization..."
    PROMPT_PLOT="Modify $EXP_DIR/plot.py. Task: $TASK_BG_TEXT. Metric: $METRIC_TEXT. Requirements: 1. Draw ONE plot (e.g. Pred vs True). 2. Save as 'best_result.png'. 3. Function signature: 'draw_plots(X_test, y_test, y_pred, output_dir, params)'."
    
    printf "%b" "$PROMPT_PLOT" | python3 "$AGENT_APP_ROOT/scripts/llm_runner.py" \
        --model "$LLM_MODEL" \
        --cwd "$EXP_DIR" \
        --whitelist "plot.py" \
        --timeout 300 \
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
        echo "❌ ERROR: strategy.py is missing 'def load_trained_model'"
        HAS_ERROR=1
    fi

    # 8. Check Evaluator Anti-Cheating Protection
    if ! grep -q "def check_data_leakage" "$EXP_DIR/evaluator.py"; then
        echo "❌ ERROR: evaluator.py is missing 'def check_data_leakage'! Security protection was removed by AI."
        HAS_ERROR=1
    fi
    if ! grep -q "check_data_leakage(" "$EXP_DIR/evaluator.py"; then
        echo "❌ ERROR: evaluator.py defines but NEVER CALLS 'check_data_leakage'! Anti-cheating is inactive."
        HAS_ERROR=1
    fi

    # 9. Dry Run (Sequential)
    echo "[Validation] performing dry run (Train -> Eval)..."
    
    # Use subshell to isolate directory change and prevent path corruption
    (
        cd "$EXP_DIR" || exit 1
        
        # 1. Run Training
        echo "Running Strategy (Training)..."
        if ! "$PYTHON_PATH" strategy.py > train_log.txt 2>&1; then
            echo "❌ Training Failed."
            cat train_log.txt | tail -n 20
            exit 101
        else
            echo "✅ Training Complete."
        fi
        
        # 2. Run Evaluation
        echo "Running Evaluator..."
        if ! "$PYTHON_PATH" evaluator.py > eval_out.txt 2>&1; then
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
    PLOT_OUTPUT=$("$PYTHON_PATH" "evaluator.py" --dry-run-plot 2>/dev/null | tr -d '[:space:]')
    cd "$current_dir" || exit
fi

# Export vars for Python script
export PROJECT_NAME
export PYTHON_PATH
export PLOT_OUTPUT
export TASK_BG_TEXT
export METRIC_TEXT

python3 - <<'EOF'
import json
import os

config_path = 'config.json'
try:
    data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            try: data = json.load(f)
            except: data = {}
    elif os.path.exists('config_template.json'):
        with open('config_template.json', 'r') as f:
            try: data = json.load(f)
            except: data = {}
    
    if 'global_vars' not in data: data['global_vars'] = {}

    # Update Fields
    # Using os.environ to get exported bash variables
    data['root_dir'] = f"./{os.environ['PROJECT_NAME']}"
    data['global_vars']['venv'] = os.environ['VENV_PYTHON']
    
    # Construct sequential eval_cmd for Agent usage (Train then Eval)
    py = os.environ['VENV_PYTHON']
    data['global_vars']['eval_cmd'] = f"{py} strategy.py && {py} evaluator.py"
    
    # Plot names with fallback
    plot_out = os.environ.get('PLOT_OUTPUT', '')
    if not plot_out:
        plot_out = "@best_result.png"
    data['global_vars']['plot_names'] = plot_out
    
    task = os.environ.get('TASK_BG_TEXT', '')
    metric = os.environ.get('METRIC_TEXT', '')
    
    # Enhanced System Prompt
    sys_instruction = (
        "1. You can improve by modifying Model Architecture (scale up/down), Hyperparameter Search (optimize search space), "
        "and Training Process (epochs, optimizer, schedule, etc).\n"
        "2. Add debug info and SAVE worst samples/predictions as .npy files for later analysis of failure cases.\n"
        "3. Optimize for speed; avoid redundancy."
    )
    
    sys_prompt = f"You are an expert AI Data Scientist. Task: {task}. Metric: {metric}. Goal: Optimize strategy.py. \n{sys_instruction}"
    data['global_vars']['DEFAULT_SYS'] = sys_prompt

    with open(config_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ config.json updated successfully. Root: {data['root_dir']}")
except Exception as e:
    print(f"❌ Failed to update config.json: {e}")
EOF

echo "Done."