import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

def split_data(data_dir, ratio=0.05):
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "Y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Error: Data files (X.npy, Y.npy) not found in directory: {data_dir}")
        return

    print(f"Loading X and Y from {data_dir}...")
    X = np.load(x_path)
    Y = np.load(y_path)
    
    # --- Data Validation ---
    print("Validating data...")
    errors = []
    if X.size == 0 or Y.size == 0:
        errors.append("Dataset is empty.")
    if X.shape[0] != Y.shape[0]:
        errors.append(f"Sample count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}.")
    
    if np.issubdtype(X.dtype, np.number) and not np.all(np.isfinite(X)):
        errors.append("X contains NaN or Inf values.")
    if np.issubdtype(Y.dtype, np.number) and not np.all(np.isfinite(Y)):
        errors.append("Y contains NaN or Inf values.")
            
    if errors:
        print("\n--- DATA CHECK FAILED ---")
        for err in errors:
            print(f"ERROR: {err}")
        sys.exit(1)
        
    print("Data validation passed.")
    print(f"Splitting data with reserved ratio: {ratio:.2%}")
    
    # Split into Development (Train/Test later) and Reserved (Holdout)
    X_dev, X_reserved, Y_dev, Y_reserved = train_test_split(
        X, Y, test_size=ratio, random_state=42
    )
    
    print(f"Development Set: {X_dev.shape[0]} samples")
    print(f"Reserved Set:    {X_reserved.shape[0]} samples")
    
    # Save files back to the same data_dir
    paths = {
        "X_dev.npy": X_dev,
        "Y_dev.npy": Y_dev,
        "X_reserved.npy": X_reserved,
        "Y_reserved.npy": Y_reserved
    }
    
    for filename, data_array in paths.items():
        path = os.path.join(data_dir, filename)
        print(f"Saving {filename} to {path}...")
        np.save(path, data_array)
        
    print("Split complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_data.py <DATA_DIR> [RESERVED_RATIO]")
        print("Example: python split_data.py ./data 0.05")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    
    # Handle ratio argument
    reserved_ratio = 0.05
    if len(sys.argv) >= 3:
        try:
            reserved_ratio = float(sys.argv[2])
            if reserved_ratio > 1:
                reserved_ratio /= 100.0
        except ValueError:
            print(f"Warning: Invalid ratio provided. Using default {reserved_ratio}")

    split_data(target_dir, reserved_ratio)
