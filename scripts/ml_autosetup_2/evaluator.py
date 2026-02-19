import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 1. Import Factory from Strategy
from strategy import load_trained_model

# 2. Import Fixed Protocol
from experiment_setup import (
    load_and_split_data, get_validation_noise_generator, PROTOCOL_SEED
)

# 3. Import Plotting
try:
    import plot
except ImportError:
    print("Warning: plot.py not found. Plotting disabled.")
    plot = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_fast.pt"
OUTPUT_DIR = os.getcwd()
BATCH_SIZE = 256

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Simple Dataset for Evaluation
class EvalDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = torch.from_numpy(X).float()
        if Y is not None:
            self.Y = torch.from_numpy(Y).float()
        else:
            self.Y = torch.zeros(len(X)) # Dummy
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

def run_inference(model, X, Y, device, desc="Inference"):
    dataset = EvalDataset(X, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    preds = []
    model.eval()
    
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            # Forward pass
            # Note: Model is expected to return (B, L, 2) or similar
            out = model(xb)
            
            # Post-process: Assume output is unit vector or phase directly
            # For this Ref, we assume output is (B, L, 2) [cos, sin]
            # We convert to phase angle for metric
            if out.shape[-1] == 2:
                # Normalize
                out = out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8)
                phi = torch.atan2(out[..., 1], out[..., 0])
            else:
                # Assume direct phase output
                phi = out.squeeze(-1)
                
            preds.append(phi.cpu().numpy())
            
    return np.concatenate(preds, axis=0)

def check_data_leakage(model, X, y_true, device):
    """
    Anti-Cheating Mechanism:
    Ensures that the model's predictions do not depend on the target labels 'y'.
    We run inference twice: once with real y, once with shuffled/random y.
    If predictions change, it means the model is peeking at y.
    """
    print("Running Data Leakage Check...")
    
    # 1. Baseline Prediction
    preds_1 = run_inference(model, X, y_true, device, desc="Leakage Check (Real Y)")
    
    # 2. Corrupted Y Prediction
    # Create a corrupted Y (Shuffle)
    y_corrupt = y_true.copy()
    np.random.shuffle(y_corrupt)
    # Ensure it's actually different
    if np.array_equal(y_corrupt, y_true):
        y_corrupt = np.random.randn(*y_true.shape).astype(np.float32)
        
    preds_2 = run_inference(model, X, y_corrupt, device, desc="Leakage Check (Fake Y)")
    
    # 3. Compare
    # Allow small float error
    if not np.allclose(preds_1, preds_2, atol=1e-5):
        diff = np.abs(preds_1 - preds_2).max()
        print(f"❌ CRITICAL: DATA LEAKAGE DETECTED! Max diff: {diff}")
        print("Model predictions changed when Target Y was modified.")
        print("This implies the model is using the Target Y during inference.")
        return True # Leaked
    
    print("✅ Pass: No data leakage detected.")
    return False

def evaluate():
    print("--- EVALUATOR STARTED ---")
    set_seed(PROTOCOL_SEED)
    
    # 1. Load Data
    _, X_val_clean, _, y_val_phi = load_and_split_data()
    N_val, P, L = X_val_clean.shape
    print(f"Validation set size: {N_val}, P={P}, L={L}")
    
    # 2. Add Noise (Protocol)
    print("Adding noise to validation set (Protocol Standard)...")
    add_noise = get_validation_noise_generator(seed=999)
    X_val_flat = X_val_clean.reshape(N_val, -1)
    X_val_noisy = add_noise(X_val_flat).reshape(N_val, P, L).astype(np.float32)
    
    # 3. Load Model using Factory
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint {CHECKPOINT_PATH} not found. Did strategy.py run?")
        
    print(f"Loading model from {CHECKPOINT_PATH}...")
    try:
        model = load_trained_model(CHECKPOINT_PATH, DEVICE)
    except Exception as e:
        print(f"Failed to load model via strategy.load_trained_model: {e}")
        raise e

    # 4. Leakage Check
    if check_data_leakage(model, X_val_noisy[:100], y_val_phi[:100], DEVICE): # Check on subset
        print("Evaluation aborted due to leakage.")
        # We assume leakage = infinite error
        print(f"Best metric: inf") 
        return

    # 5. Full Inference
    print("Running full inference...")
    start_t = os.times().elapsed
    preds_phi = run_inference(model, X_val_noisy, y_val_phi, DEVICE)
    
    # 6. Metric (Wrapped MAE)
    # Wrap difference to [-pi, pi]
    diff = preds_phi - y_val_phi
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    mae = np.mean(np.abs(diff))
    
    print(f"=================================")
    print(f"FINAL EVALUATION SCORE (Wrapped MAE): {mae:.6f} rad")
    print(f"Best metric: {mae:.6f}") # Required format
    print(f"=================================")
    
    with open("eval_out.txt", "w") as f:
        f.write(f"Final Score (Wrapped Phase MAE): {mae:.6f}\n")

    # 7. Plotting
    if plot:
        print("Generating plots...")
        try:
            params_dummy = {"add_noise": True}
            y_val_cs = np.stack([np.cos(y_val_phi), np.sin(y_val_phi)], axis=-1)
            y_val_cs_pred = np.stack([np.cos(preds_phi), np.sin(preds_phi)], axis=-1)
            
            # Check signature of draw_plots
            # Adapt input if necessary
            plot.draw_plots(X_val_noisy, y_val_cs, y_val_cs_pred, OUTPUT_DIR, params_dummy)
            print("Plots generated.")
        except Exception as e:
            print(f"Plotting failed: {e}")
            # Don't crash eval for plotting error

if __name__ == "__main__":
    evaluate()
