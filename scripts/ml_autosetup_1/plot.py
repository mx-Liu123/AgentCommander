import matplotlib
# Force non-interactive backend to prevent window popup
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import numpy as np

def draw_plots(X_test, y_test, y_pred, output_dir, params):
    """
    Draws and saves visualization for the current best model.
    
    Args:
        X_test (np.array): Test features.
        y_test (np.array): Ground truth labels.
        y_pred (np.array): Model predictions.
        output_dir (str): Directory to save plots.
        params (dict): Parameters of the best model.
        
    Returns:
        list: List of saved filenames (absolute or relative paths).
    """
    # Define filename - static name to overwrite previous best
    filename = "best_result_plot.png"
    filepath = os.path.join(output_dir, filename)
    
    try:
        plt.figure(figsize=(10, 6))
        
        # Simple default visualization: First 100 samples comparison
        # (Works for both Regression and some Classification logic)
        limit = min(100, len(y_test))
        indices = np.arange(limit)
        
        # Flatten if necessary
        y_t = y_test.ravel()[:limit]
        y_p = y_pred.ravel()[:limit]
        
        plt.plot(indices, y_t, label="Ground Truth", marker='o', alpha=0.7)
        plt.plot(indices, y_p, label="Prediction", marker='x', alpha=0.7)
        
        plt.title(f"Best Model Performance (First {limit} samples)\nScore Improved")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close() # Close memory to avoid leaks
        
        return [filename] # Return the filename
        
    except Exception as e:
        print(f"Warning: Plotting failed inside plot.py: {e}")
        return []
