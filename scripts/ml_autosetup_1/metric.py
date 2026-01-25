import numpy as np

# --- Metric Configuration ---
# MSE: Lower is better.
HIGHER_IS_BETTER = False 

def calculate_score(y_true, y_pred):
    """
    Calculates the Normalized Mean Squared Error (NMSE) per sample, then averages.
    NMSE = sum((true - pred)^2) / sum(true^2)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    epsilon = 1e-8
    
    # Flatten last dimensions, keep batch dimension
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    error_energy = np.sum((y_true_flat - y_pred_flat) ** 2, axis=1)
    true_energy = np.sum(y_true_flat ** 2, axis=1)
    
    nmse_per_sample = error_energy / (true_energy + epsilon)
    
    return np.mean(nmse_per_sample)
