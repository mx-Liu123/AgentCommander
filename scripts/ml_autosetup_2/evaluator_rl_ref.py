import os
import sys
import numpy as np
import pandas as pd

# 1. Contracted Interfaces
try:
    from strategy import load_trained_model
except ImportError:
    def load_trained_model(path, device): raise NotImplementedError("strategy.py must implement load_trained_model.")

import metric
import plot

def create_env():
    """
    Discovery: Tries to load the environment from strategy_lib.
    """
    try:
        from strategy_lib.env import NautilusEnv
        return NautilusEnv()
    except Exception as e:
        print(f"Standard environment loading failed: {e}")
        # Fallback for other RL tasks
        return None

def evaluate():
    print("--- STANDARDIZED RL EVALUATOR ENGINE ---")
    
    # Setup
    env = create_env()
    if env is None:
        print("❌ Error: Could not initialize environment.")
        return

    # Checkpoint selection
    checkpoint_path = "best_model.zip"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "best_model.pt"
        
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Model checkpoint not found.")
        return

    model = load_trained_model(checkpoint_path, "cpu")
    
    # 2. Universal Interaction Loop
    print("🚀 Running Interaction Loop...")
    history = []
    obs, _ = env.reset()
    done = False
    
    while not done:
        # Standard SB3-style or custom predict interface
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Standardize record
        info['reward_step'] = reward
        history.append(info)
        
        if terminated or truncated:
            break

    # 3. Delegation to Outsourced Brain (Metric)
    print("📊 Outsourcing score calculation to metric.py...")
    final_score = metric.calculate_rl_score(history)
    
    print(f"=================================")
    print(f"Best metric: {final_score:.6f}") 
    print(f"=================================")

    # 4. Delegation to Outsourced Eyes (Plot)
    print("🎨 Outsourcing visualization to plot.py...")
    try:
        plot.draw_rl_plots(history, "plots_eod")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    if "--dry-run-plot" in sys.argv:
        # Standard anchor for setup script to find the plot name
        print("@best_result.png")
        sys.exit(0)
    evaluate()
