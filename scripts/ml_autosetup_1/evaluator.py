import sys
import os
import time
import importlib.util
import csv
import traceback
import numpy as np
from sklearn.model_selection import train_test_split

# --- Config & Setup ---
OUTPUT_DIR = os.getcwd()

# Data Configuration
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
X_PATH = os.path.join(DATA_DIR, "X_dev.npy")
Y_PATH = os.path.join(DATA_DIR, "Y_dev.npy")
TEST_SIZE = 0.2  # Fraction of dev set used for internal testing
RANDOM_SEED = 42

# Metric Configuration
import metric  # Assumes metric.py is in the same directory

# Optional Plotting Configuration
try:
    import plot
except ImportError:
    plot = None

# 0. Basic Config Validation
if not (0 < TEST_SIZE < 1):
    raise ValueError(f"TEST_SIZE must be between 0 and 1, got {TEST_SIZE}")

if not hasattr(metric, 'calculate_score') or not hasattr(metric, 'HIGHER_IS_BETTER'):
    raise AttributeError("metric.py must define 'calculate_score(y_true, y_pred)' and 'HIGHER_IS_BETTER' (bool).")

def evaluate_single_config(strategy_class, params, X_train, X_test, y_train, y_test):
    """
    Runs a single training and evaluation cycle. 
    Returns a score where LOWER is always BETTER.
    """
    try:
        model = strategy_class(params=params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Check prediction shape
        if predictions.shape != y_test.shape:
             raise ValueError(f"Prediction shape mismatch. Expected {y_test.shape}, got {predictions.shape}")
             
        raw_score = metric.calculate_score(y_test, predictions)
        
        # Standardize to "Lower is Better"
        if metric.HIGHER_IS_BETTER:
            return -raw_score
        return raw_score
    except Exception as e:
        print(f"Error with params {params}: {e}")
        return float('inf')

def run_parameter_search(strategy_module):
    """
    Orchestrates the parameter search. Always MINIMIZES the score.
    """
    # 0. Check if OUTPUT_DIR is writable
    if not os.access(OUTPUT_DIR, os.W_OK):
        raise PermissionError(f"Output directory {OUTPUT_DIR} is not writable.")

    # 1. Load Data
    print(f"Loading data from {DATA_DIR}...")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Data files not found. Please ensure {X_PATH} and {Y_PATH} exist.")
        
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    # --- Data Validation ---
    print("Validating data...")
    errors = []
    if X.size == 0 or y.size == 0:
        errors.append("Dataset is empty.")
    if X.shape[0] != y.shape[0]:
        errors.append(f"Sample count mismatch: X has {X.shape[0]}, y has {y.shape[0]}.")
    
    if np.issubdtype(X.dtype, np.number) and not np.all(np.isfinite(X)):
        errors.append("X contains NaN or Inf values.")
    if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
        errors.append("y contains NaN or Inf values.")
        
    if errors:
        print("\n--- DATA CHECK FAILED ---")
        for err in errors:
            print(f"ERROR: {err}")
        raise ValueError("Data validation failed. Search aborted.")
    print("Data validation passed.")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"Data Loaded: Train shape {X_train.shape}, Test shape {X_test.shape}")
    
    # 3. Get Configs
    if hasattr(strategy_module, 'get_search_configs'):
        configs = strategy_module.get_search_configs()
    else:
        configs = [{}]

    # --- Fail-Fast Leakage Check ---
    # Check the first configuration before spending time on search
    if configs:
        print("\n--- Performing Fail-Fast Leakage Check ---")
        if check_data_leakage(strategy_module.Strategy, configs[0], X_train, X_test, y_train, y_test):
            print("ERROR: Data leakage detected in initial check. Aborting search.")
            return float('inf')

    print(f"\nStarting search (LOWER score is BETTER)...")
    
    soft_limit_seconds = 10 * 60 
    hard_limit_seconds = 15 * 60 
    
    start_time = time.time()
    soft_limit_end = start_time + soft_limit_seconds
    hard_limit_end = start_time + hard_limit_seconds
    
    best_score = float('inf')
    best_params = None
    
    results_file = os.path.join(OUTPUT_DIR, "search_results.csv")
    fieldnames = ["run_id", "score", "params"]
    
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    for i, params in enumerate(configs):
        current_time = time.time()
        if current_time > soft_limit_end:
            break
            
        try:
            if current_time > hard_limit_end:
                 raise TimeoutError("Hard time limit reached.")

            score = evaluate_single_config(strategy_module.Strategy, params, X_train, X_test, y_train, y_test)
            
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({"run_id": i+1, "score": score, "params": str(params)})

            # Always minimize
            if score < best_score:
                best_score = score
                best_params = params
                print(f"New Best [Run {i+1}]: Score={score:.4f} (Lower is better)")
                
                # --- Visualization Hook ---
                if plot is not None and hasattr(plot, 'draw_plots'):
                    try:
                        # Re-run prediction for plotting (Minimal change to structure)
                        # We do this only for the 'Best' so cost is acceptable.
                        best_model = strategy_module.Strategy(params=params)
                        best_model.fit(X_train, y_train)
                        best_preds = best_model.predict(X_test)
                        
                        plot_files = plot.draw_plots(X_test, y_test, best_preds, OUTPUT_DIR, params)
                        if plot_files:
                            # Format for LLM/User: @file1.png,@file2.png
                            formatted_names = ",".join([f"@{os.path.basename(f)}" for f in plot_files])
                            print(f"PLOT_GENERATED: {formatted_names}")
                    except Exception as e:
                        print(f"Warning: Visualization failed: {e}")

            elif (i + 1) % 10 == 0:
                 print(f"Processed {i+1}/{len(configs)}...")
        
        except TimeoutError:
            break
        except Exception:
            traceback.print_exc()
            continue
             
    print(f"\nSearch Complete. Best Score: {best_score}")
    
    # --- Final Data Leakage Check ---
    if best_params is not None:
        print("\n--- Running Final Data Leakage Check on Best Config ---")
        if check_data_leakage(strategy_module.Strategy, best_params, X_train, X_test, y_train, y_test):
            print("CRITICAL: Best configuration failed leakage check! Invalidating score.")
            return float('inf')

    return best_score

def check_data_leakage(strategy_class, params, X_train, X_test, y_train, y_test):
    """
    Returns True if leakage detected, False otherwise.
    """
    print("Checking for memory leakage...")
    has_leakage = False
    
    try:
        # 1. Train Control Model
        model = strategy_class(params=params)
        model.fit(X_train, y_train)
        preds_control = model.predict(X_test)

        # 2. Modify y_test in memory
        y_test_backup = y_test.copy()
        noise = y_test.copy()
        np.random.shuffle(noise)
        
        if np.array_equal(noise, y_test) and noise.size > 0:
            if np.issubdtype(noise.dtype, np.number):
                noise.flat[0] += 1
            else:
                if noise.size > 1:
                     val = noise.flat[0]
                     noise.flat[0] = noise.flat[1]
                     noise.flat[1] = val
        
        np.copyto(y_test, noise)
        
        # 3. Predict again
        preds_exp = model.predict(X_test)
        
        # 4. Restore y_test
        np.copyto(y_test, y_test_backup)
        
        # 5. Compare
        if np.issubdtype(preds_control.dtype, np.number) and np.issubdtype(preds_exp.dtype, np.number):
             has_change = not np.allclose(preds_control, preds_exp, equal_nan=True)
        else:
             has_change = not np.array_equal(preds_control, preds_exp)
        
        if has_change:
            print(f"CRITICAL WARNING: MEMORY DATA LEAKAGE DETECTED!")
            has_leakage = True
        else:
            print("Pass: No memory leakage detected.")
            
    except Exception as e:
        print(f"Leakage check failed with error: {e}")
    finally:
        if 'y_test_backup' in locals():
             np.copyto(y_test, y_test_backup)
             
    return has_leakage
def dry_run_plot_names():
    """
    Executes a dummy plot generation to retrieve filenames.
    Prints ONLY the formatted list: @file1.png,@file2.png
    """
    try:
        import plot
        if not hasattr(plot, 'draw_plots'):
            return 
            
        # Create dummy data
        X_dummy = np.zeros((5, 5))
        y_dummy = np.zeros((5,))
        y_pred_dummy = np.zeros((5,))
        params_dummy = {}
        
        # Call with current directory as output
        files = plot.draw_plots(X_dummy, y_dummy, y_pred_dummy, ".", params_dummy)
        
        if files:
            formatted = ",".join([f"@{os.path.basename(f)}" for f in files])
            print(formatted)
            
    except Exception:
        pass

def evaluate(strategy_path):
    """
    Main entry point invoked by the agent/user command.
    """
    # Check if file exists
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")

    # Dynamic import
    file_path = os.path.abspath(strategy_path)
    module_name = "user_strategy_" + str(int(time.time())) # Unique name to avoid caching issues if run repeatedly
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy from {strategy_path}")
        
    strategy_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = strategy_module
    spec.loader.exec_module(strategy_module)
    
    # Check for Strategy class and required methods
    if not hasattr(strategy_module, 'Strategy'):
        raise AttributeError(f"Module {strategy_path} must define a 'Strategy' class.")
    
    # Check for required methods on Strategy
    for method in ['fit', 'predict']:
        if not hasattr(strategy_module.Strategy, method):
            raise AttributeError(f"Strategy class in {strategy_path} is missing required method: '{method}'")
        
    return run_parameter_search(strategy_module)

if __name__ == "__main__":
    if "--dry-run-plot" in sys.argv:
        dry_run_plot_names()
        sys.exit(0)

    if len(sys.argv) > 1:
        print(evaluate(sys.argv[1]))
    else:
        print("Usage: python evaluator.py <path_to_strategy.py>")
