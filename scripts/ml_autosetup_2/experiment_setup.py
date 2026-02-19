import os
import numpy as np

# ==========================================================
# FIXED PROTOCOL CONSTANTS
# (Agent should NOT modify these)
# ==========================================================
PROTOCOL_SEED = 42
VAL_SPLIT_RATIO = 0.15

DATA_PATH = "~/test_data/"
NPZ_NAME = "lr_signals_with_params_E_B_phase_base.npz"

# Validation noise settings (The "Exam" difficulty)
VAL_SNR_LO = 20
VAL_SNR_HI = 30

def get_protocol_seed():
    return PROTOCOL_SEED

def load_and_split_data():
    """
    Standardized data loading and splitting protocol.
    Guarantees that Strategy and Evaluator always see the same data splits.
    """
    full_path = os.path.join(DATA_PATH, NPZ_NAME)
    
    # Fallback for dry-run/local testing
    if not os.path.exists(full_path):
        if os.path.exists(NPZ_NAME): 
            full_path = NPZ_NAME
        else:
            print(f"[Protocol] Warning: Data file {full_path} not found. Using dummy random data for dry run.")
            P, R, L = 2, 100, 1000
            X_B = np.random.randn(P, R, L)
            phase_B = np.random.randn(R, L)
            return _split_arrays(X_B, phase_B, R)

    print(f"[Protocol] Loading data from {full_path}...")
    data = np.load(full_path, allow_pickle=True)
    X_B = data["X_B"]         # (P, R, L)
    phase_B = data["phase_B"] # (R, L)
    P, R, L = X_B.shape
    return _split_arrays(X_B, phase_B, R)

def _split_arrays(X_B, phase_B, R):
    # 1. Fixed Preprocessing: Center data along time
    X_centered = X_B - X_B.mean(axis=2, keepdims=True)
    
    # 2. Fixed Splitting
    rg = np.random.default_rng(PROTOCOL_SEED)
    idx = np.arange(R)
    rg.shuffle(idx)
    
    n_val = int(round(VAL_SPLIT_RATIO * R))
    
    val_idx = np.sort(idx[:n_val])
    trn_idx = np.sort(idx[n_val:])
    
    # 3. Create Arrays
    X_tr = np.transpose(X_centered[:, trn_idx, :], (1, 0, 2)).astype(np.float32)
    phi_tr = phase_B[trn_idx].astype(np.float32)
    
    X_va = np.transpose(X_centered[:, val_idx, :], (1, 0, 2)).astype(np.float32)
    phi_va = phase_B[val_idx].astype(np.float32)
    
    print(f"[Protocol] Data Split: Train={len(trn_idx)}, Val={len(val_idx)} (Split={VAL_SPLIT_RATIO:.2f})")
    
    return X_tr, X_va, phi_tr, phi_va

def get_validation_noise_generator(seed=999):
    """
    Returns a function to add noise consistent with the evaluation protocol.
    Used by Evaluator.
    """
    def add_noise(X_flat):
        rg = np.random.default_rng(seed)
        snrs = rg.integers(VAL_SNR_LO, VAL_SNR_HI, size=X_flat.shape[0])
        s_x  = np.sqrt(np.sum(X_flat**2, axis=1))
        s_x = np.maximum(s_x, 1e-9)
        ns   = (s_x / snrs)[:, None]
        return X_flat + rg.normal(0.0, ns, size=X_flat.shape)
    return add_noise
