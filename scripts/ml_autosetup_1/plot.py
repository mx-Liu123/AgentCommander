import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

def draw_plots(X_test, y_test, y_pred, output_dir, params):
    """
    Draws 3 random samples comparing Ground Truth Phase vs Predicted Phase.
    """
    filename = "best_result.png"
    filepath = os.path.join(output_dir, filename)
    
    try:
        num_samples = len(X_test)
        
        # Calculate Normalized MSE (NMSE) for each sample individually
        epsilon = 1e-8
        y_true_flat = y_test.reshape(num_samples, -1)
        y_pred_flat = y_pred.reshape(num_samples, -1)

        error_energy = np.sum((y_true_flat - y_pred_flat) ** 2, axis=1)
        true_energy = np.sum(y_true_flat ** 2, axis=1)
        
        sample_nmses = error_energy / (true_energy + epsilon)
        
        # Get indices of the 3 samples with the HIGHEST error (Worst first)
        k = min(3, num_samples)
        # argsort gives ascending order, so we take the last k elements and reverse them
        worst_indices = np.argsort(sample_nmses)[-k:][::-1]
        
        indices = worst_indices
        
        fig, axes = plt.subplots(len(indices), 1, figsize=(10, 3 * len(indices)), squeeze=False)
        
        for i, idx in enumerate(indices):
            ax = axes[i, 0]
            
            # Handle dimensions: Ensure 1D array for plotting
            y_true_sample = y_test[idx].ravel()
            y_pred_sample = y_pred[idx].ravel()
            
            ax.plot(y_true_sample, label="Ground Truth", color='black', alpha=0.8, linewidth=1.5)
            ax.plot(y_pred_sample, label="Prediction", color='cyan', linestyle='--', alpha=0.9, linewidth=1.5)
            
            ax.set_title(f"Sample {idx}: Phase Reconstruction (NMSE: {sample_nmses[idx]:.4f})")
            ax.set_ylabel("Phase")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            
            if i == len(indices) - 1:
                ax.set_xlabel("Time Step")

        plt.suptitle("Gravitational Wave Phase Prediction (Worst 3 Samples by NMSE)", fontsize=14)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return [filename]
        
    except Exception as e:
        print(f"Plotting error: {e}")
        return []
