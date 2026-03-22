import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 1. Contracted Interfaces
from strategy import load_trained_model
from experiment_setup import load_and_split_data, PROTOCOL_SEED
import metric
import plot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_fast.pt"
BATCH_SIZE = 256

def evaluate():
    print("--- STANDARDIZED ML EVALUATOR ENGINE ---")
    
    # 1. Load Data
    _, X_val, _, y_val = load_and_split_data()
    print(f"Validation set size: {len(X_val)}")
    
    # 2. Load Model
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint {CHECKPOINT_PATH} not found.")
    
    model = load_trained_model(CHECKPOINT_PATH, DEVICE)
    model.eval()

    # 3. Standard Batch Inference
    X_tensor = torch.from_numpy(X_val).float().to(DEVICE)
    y_preds = []
    
    with torch.no_grad():
        # Simple split for memory safety
        for i in range(0, len(X_tensor), BATCH_SIZE):
            batch = X_tensor[i:i+BATCH_SIZE]
            out = model(batch)
            y_preds.append(out.cpu().numpy())
            
    y_pred_final = np.concatenate(y_preds, axis=0)

    # 4. Delegation to Outsourced Brain (Metric)
    print("📊 Outsourcing score calculation to metric.py...")
    final_score = metric.calculate_standard_score(y_val, y_pred_final)
    
    print(f"=================================")
    print(f"Best metric: {final_score:.6f}") 
    print(f"=================================")

    # 5. Delegation to Outsourced Eyes (Plot)
    print("🎨 Outsourcing visualization to plot.py...")
    try:
        plot.draw_standard_plots(X_val, y_val, y_pred_final, "plots_eod")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    if "--dry-run-plot" in sys.argv:
        print("@best_result.png")
        sys.exit(0)
    evaluate()
