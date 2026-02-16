#!/bin/bash

# <GEMINI_UI_CONFIG>
# {
#   "name": "ML Auto-Setup (Standard)",
#   "description": "Full-stack ML experiment setup: Data splitting, Strategy/Metric generation via AI, and initial evaluation.",
#   "inputs": [
#     {"id": "PROJECT_NAME", "label": "Project Name", "type": "text", "default": "my_new_experiment", "tooltip": "Folder name to create"},
#     {"id": "DATA_DIR", "label": "Data Directory", "type": "text", "default": "/home/liumx/data/diabetes", "tooltip": "Absolute path containing X.npy and Y.npy"},
#     {"id": "VENV_PYTHON", "label": "Python Interpreter", "type": "text", "default": "/home/liumx/.conda/envs/agent_commander/bin/python"},
#     {"id": "RESERVED_RATIO", "label": "Reserved Data Ratio", "type": "number", "default": 0.05},
#     {"id": "TEST_SET_RATIO", "label": "Test Set Ratio", "type": "number", "default": 0.2},
#     {"id": "SOFT_LIMIT", "label": "Soft Time Limit (s)", "type": "number", "default": 600},
#     {"id": "HARD_LIMIT", "label": "Hard Time Limit (s)", "type": "number", "default": 900},
#     {"id": "USER_SEED", "label": "Random Seed", "type": "text", "default": "42", "tooltip": "Enter a number or 'random'"},
#     {"id": "METRIC_TEXT", "label": "Metric Description", "type": "textarea", "default": "Use sklearn.metrics.accuracy_score.", "rows": 2},
#     {"id": "TASK_BG_TEXT", "label": "Task Background", "type": "textarea", "default": "Tabular classification.", "rows": 2},
#     {"id": "MODEL_HINT_TEXT", "label": "Model Hints", "type": "textarea", "default": "Suggest using Random Forest or XGBoost.", "rows": 2}
#   ],
#   "preview_steps": [
#     "1. Environment Check & Confirmation",
#     "2. Create Directory Structure",
#     "3. Data Splitting",
#     "4. AI Code Generation (Strategy, Metric, Plot)",
#     "5. Validation & Dry Run",
#     "6. Update config.json"
#   ]
# }
# </GEMINI_UI_CONFIG>

# 1. Locate Source Files (Assumes script is in the same dir as templates)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SPLIT_SCRIPT="$SCRIPT_DIR/split_data.py"
EVALUATOR_SCRIPT="$SCRIPT_DIR/evaluator.py"
STRATEGY_SCRIPT="$SCRIPT_DIR/strategy.py"
METRIC_SCRIPT="$SCRIPT_DIR/metric.py"
PLOT_SCRIPT="$SCRIPT_DIR/plot.py"

# Check if source files exist
for file in "$SPLIT_SCRIPT" "$EVALUATOR_SCRIPT" "$STRATEGY_SCRIPT" "$METRIC_SCRIPT" "$PLOT_SCRIPT"; do
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
    echo "cd {current_exp_path} && {venv} -c \"from evaluator import evaluate; print('Best metric:', evaluate('strategy.py'))\""
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
        if [ -n "$default" ]; then
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
get_input "VENV_PYTHON" "Python Interpreter Path" "$DEFAULT_VENV"

DEFAULT_RESERVED=0.05
get_input "RESERVED_RATIO" "Reserved Data Ratio (0-1)" "$DEFAULT_RESERVED"

echo -e "\n--- Evaluation Config ---"
DEFAULT_TEST_RATIO=0.2
get_input "TEST_SET_RATIO" "Internal Test Set Ratio (0-1)" "$DEFAULT_TEST_RATIO"

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
EXP_DIR="$PROJECT_ROOT/Branch_example/exp_example"

mkdir -p "$EXP_DIR/data" # Create data dir placeholder, though we use absolute path
cp "$SPLIT_SCRIPT" "$PROJECT_ROOT/"
cp "$EVALUATOR_SCRIPT" "$STRATEGY_SCRIPT" "$METRIC_SCRIPT" "$PLOT_SCRIPT" "$EXP_DIR/"

echo "[Setup] Files copied to $EXP_DIR"

# ==============================================================================
# 4. Data Preparation
# ==============================================================================

echo -e "\n[Data] Running split_data.py on $DATA_DIR..."
# We use the system python or venv python to run the split script? Use VENV for safety
"$VENV_PYTHON" "$PROJECT_ROOT/split_data.py" "$DATA_DIR" "$RESERVED_RATIO"

if [ $? -ne 0 ]; then
    echo "Error: Data splitting failed."
    exit 1
fi

# ==============================================================================
# 5. Configure Evaluator (sed)
# ==============================================================================

TARGET_EVAL="$EXP_DIR/evaluator.py"
echo "[Config] Injecting settings into $TARGET_EVAL..."

# 1. Update DATA_DIR (Using | as delimiter to handle paths)
sed -i "s|^DATA_DIR = .*|DATA_DIR = \"$DATA_DIR\"|" "$TARGET_EVAL"

# 2. Update TEST_SIZE
sed -i "s/^TEST_SIZE = .*/TEST_SIZE = $TEST_SET_RATIO/" "$TARGET_EVAL"

# 3. Update Time Limits
sed -i "s/soft_limit_seconds = .*/soft_limit_seconds = $SOFT_LIMIT/" "$TARGET_EVAL"
sed -i "s/hard_limit_seconds = .*/hard_limit_seconds = $HARD_LIMIT/" "$TARGET_EVAL"

# 4. Update Random Seed
sed -i "s/^RANDOM_SEED = .*/RANDOM_SEED = $RANDOM_SEED/" "$TARGET_EVAL"

# --- Baseline Integrity Check ---
if command -v md5sum &> /dev/null; then
    EVAL_HASH_START=$(md5sum "$TARGET_EVAL" | awk '{print $1}')
else
    # Fallback for systems without md5sum
    EVAL_HASH_START="unknown"
fi

echo "Evaluator Hash (Baseline): $EVAL_HASH_START"

# ==============================================================================
# 6. AI Generation Loop
# ==============================================================================

while true; do
    echo -e "\n========================================"
    echo "   Starting AI Code Generation..."
    echo "========================================"

    # --- Step 5: Strategy Generation ---
    echo "[Gemini] Generating Strategy..."
    PROMPT_STRATEGY="解释我们正在制作一个给 Agent 的工作目录。 我们的 $EXP_DIR/evaluator.py 是个裁判， 然后选手是 $EXP_DIR/strategy.py，写一下这两个文件的结构, 然后任务背景是: $TASK_BG_TEXT。绝对不可以修改 evaluator.py。 模型的提示：$MODEL_HINT_TEXT。 现在修改模型，在 $EXP_DIR/strategy.py, 注意保留 def get_search_configs():，class Strategy: def __init__(self, params=None):，def fit(self, X, y):，def predict(self, X):"
    
    gemini -y "$PROMPT_STRATEGY"

    # --- Step 6: Metric Generation ---
    echo "[Gemini] Generating Metric..."
    PROMPT_METRIC="提示：$METRIC_TEXT， 现在修改 $EXP_DIR/metric.py"
    
    gemini -y --resume latest "$PROMPT_METRIC"
    
    # --- Step 7: Plot Generation (New) ---
    echo "[Gemini] Generating Plot Visualization..."
    PROMPT_PLOT="我们增加了一个可视化插件 $EXP_DIR/plot.py。请根据任务背景 '$TASK_BG_TEXT' 和指标 '$METRIC_TEXT' 修改这个文件。要求：1. 只画一张最能代表模型效果的图（如预测vs真实值，或混淆矩阵）。2. 图表必须清晰美观。3. 保存文件名为 best_result.png（或根据任务命名）。4. 不要调用 plt.show()，只保存。5. 函数签名必须是 draw_plots(X_test, y_test, y_pred, output_dir, params) 并返回文件名列表。"
    
    gemini -y --resume latest "$PROMPT_PLOT"

    # ==============================================================================
    # 7-9. Validation & Integrity Checks
    # ==============================================================================
    
    echo -e "\n[Validation] Checking integrity and functionality..."
    HAS_ERROR=0

    # 7. Check Strategy Structure
    REQUIRED_STRINGS=("class Strategy" "def get_search_configs" "def __init__" "def fit" "def predict")
    for req in "${REQUIRED_STRINGS[@]}"; do
        if ! grep -q "$req" "$EXP_DIR/strategy.py"; then
            echo "❌ ERROR: strategy.py is missing '$req'"
            HAS_ERROR=1
        fi
    done

    # 8. Check Evaluator Integrity
    if [ "$EVAL_HASH_START" != "unknown" ]; then
        EVAL_HASH_NOW=$(md5sum "$TARGET_EVAL" | awk '{print $1}')
        if [ "$EVAL_HASH_START" != "$EVAL_HASH_NOW" ]; then
            echo "❌ CRITICAL: evaluator.py has been modified by AI!"
            echo "Expected: $EVAL_HASH_START"
            echo "Actual:   $EVAL_HASH_NOW"
            HAS_ERROR=1
        else
             echo "✅ Evaluator integrity verified."
        fi
    fi

    # 9. Dry Run
    echo "[Validation] performing dry run..."
    cd "$EXP_DIR" || exit
    # Run a quick check using python -c. We assume evaluate returns a float (score) or raises error.
    # We use a subshell to not affect current dir
    DRY_RUN_OUT=$("$VENV_PYTHON" -c "
import sys
try:
    from evaluator import evaluate
    print('Starting Dry Run...')
    # We pass the strategy filename
    score = evaluate('strategy.py')
    print(f'Dry Run Success. Best Metric: {score}')
except Exception as e:
    print(f'Dry Run Failed: {e}')
    sys.exit(1)
" 2>&1)
    
    DRY_RUN_EXIT_CODE=$?
    cd - > /dev/null # Go back to root

    if [ $DRY_RUN_EXIT_CODE -eq 0 ]; then
        echo "✅ Dry Run Successful."
        echo "$DRY_RUN_OUT" | grep "Best Metric"
        # Check for plot output
        echo "$DRY_RUN_OUT" | grep "PLOT_GENERATED"
    else
        echo "❌ Dry Run Failed."
        echo "--- Output ---"
        echo "$DRY_RUN_OUT"
        echo "--------------"
        HAS_ERROR=1
    fi

    # ==============================================================================
    # 10. Result & Retry Prompt
    # ==============================================================================

    if [ $HAS_ERROR -eq 1 ]; then
        echo -e "\n⚠️  Errors were detected during generation or validation."
        read -p "Do you want to retry the AI generation steps (5-7)? [Y/n] " RETRY_CHOICE
        RETRY_CHOICE=${RETRY_CHOICE:-Y}
        if [[ "$RETRY_CHOICE" =~ ^[Nn]$ ]]; then
            echo "Exiting with errors."
            break
        fi
        # Loop continues
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
    PLOT_OUTPUT=$("$VENV_PYTHON" "evaluator.py" --dry-run-plot 2>/dev/null | tr -d '[:space:]')
    cd "$current_dir" || exit
fi

# Export vars for Python script
export PROJECT_NAME
export VENV_PYTHON
export PLOT_OUTPUT
export TASK_BG_TEXT
export METRIC_TEXT

python3 -c "
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
    data['root_dir'] = f\"./{os.environ['PROJECT_NAME']}\"
    data['global_vars']['venv'] = os.environ['VENV_PYTHON']
    data['global_vars']['plot_names'] = os.environ.get('PLOT_OUTPUT', '')
    
    task = os.environ.get('TASK_BG_TEXT', '')
    metric = os.environ.get('METRIC_TEXT', '')
    
    # Enhanced System Prompt
    sys_instruction = (
        "1. You can improve by modifying Model Architecture (scale up/down), Hyperparameter Search (optimize search space), "
        "and Training Process (epochs, optimizer, schedule, etc).\n"
        "2. Add debug info and SAVE worst samples/predictions as .npy files for later analysis of failure cases.\n"
        "3. Optimize for speed; avoid redundancy."
    )
    
    sys_prompt = f\"You are an expert AI Data Scientist. Task: {task}. Metric: {metric}. Goal: Optimize strategy.py. \n{sys_instruction}\"
    data['global_vars']['DEFAULT_SYS'] = sys_prompt

    with open(config_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'✅ config.json updated successfully. Root: {data[\"root_dir\"]}')
except Exception as e:
    print(f'❌ Failed to update config.json: {e}')
"

echo "Done."