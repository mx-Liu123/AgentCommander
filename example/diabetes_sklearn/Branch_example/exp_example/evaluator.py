import sys
import os
import time
import importlib.util
import csv
import traceback
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Config & Setup ---
OUTPUT_DIR = os.getcwd()

def evaluate_single_config(strategy_class, params, X_train, X_test, y_train, y_test):
    """
    Runs a single training and evaluation cycle for a given parameter set.
    """
    try:
        model = strategy_class(params=params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = mean_squared_error(y_test, predictions) # MSE: Lower is better
        return score
    except Exception as e:
        print(f"Error with params {params}: {e}")
        return float('inf')

def run_parameter_search(strategy_module):
    """
    Orchestrates the parameter search with time limits.
    """
    # 1. Load Data (Replaces crypto loading)
    data = load_diabetes()
    X = data.data
    y = data.target
    
    # 2. Split Data (Fixed seed for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Get Configs
    if hasattr(strategy_module, 'get_search_configs'):
        configs = strategy_module.get_search_configs()
    else:
        print("Strategy module does not define 'get_search_configs', using default empty dict.")
        configs = [{}]

    print(f"Starting parameter search over {len(configs)} configurations...")
    
    # --- Time Limit Configuration (Restored) ---
    soft_limit_seconds = 10 * 60 # 10 minutes
    hard_limit_seconds = 15 * 60 # 15 minutes
    
    start_time = time.time()
    soft_limit_end = start_time + soft_limit_seconds
    hard_limit_end = start_time + hard_limit_seconds
    
    print(f"Soft Time Limit: 10 min (Stop new runs). Hard Time Limit: 15 min (Kill current run).")
    
    best_score = float('inf') # Lower is better
    best_params = None
    
    results_file = os.path.join(OUTPUT_DIR, "search_results.csv")
    fieldnames = ["run_id", "accuracy", "params"]
    
    # Initialize results file
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    for i, params in enumerate(configs):
        current_time = time.time()
        
        # Soft Limit Check: If we passed the soft limit, don't start a NEW run.
        if current_time > soft_limit_end:
            print(f"Soft time limit ({soft_limit_seconds/60} min) reached. Stopping search loop.")
            break
            
        try:
            # Check Hard Limit inside the loop logic if operations were long, 
            # but for simple sklearn fit, we check before start.
            if current_time > hard_limit_end:
                 raise TimeoutError("Hard time limit reached.")

            score = evaluate_single_config(strategy_module.Strategy, params, X_train, X_test, y_train, y_test)
            
            # Log result
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({"run_id": i+1, "accuracy": score, "params": str(params)})

            # Simple progress log every 10 runs or if new best
            if score < best_score: # Lower is better
                best_score = score
                best_params = params
                print(f"New Best [Run {i+1}]: MSE={score:.4f} | Params={params}")
            elif (i + 1) % 10 == 0:
                 print(f"Processed {i+1}/{len(configs)}...")
        
        except TimeoutError as te:
            print(f"Run {i+1} aborted: {te}")
            break
        except Exception as e:
            print(f"Run {i+1} failed: {e}")
            traceback.print_exc()
            # Depending on strictness, we might want to continue or stop. 
            # Original code seemed to allow failures to pass or crash.
            continue
             
    total_time = time.time() - start_time
    print(f"\nSearch Complete in {total_time:.2f}s")
    print(f"Best metric: {best_score}")
    print(f"Best Params: {best_params}")
    
    return best_score

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
    
    # Check for Strategy class
    if not hasattr(strategy_module, 'Strategy'):
        raise AttributeError(f"Module {strategy_path} must define a 'Strategy' class.")
        
    return run_parameter_search(strategy_module)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(evaluate(sys.argv[1]))
    elif os.path.exists("strategy.py"):
        print(f"No argument provided. Defaulting to 'strategy.py'...")
        print(evaluate("strategy.py"))
    else:
        print("Usage: python evaluator.py <path_to_strategy.py>")
