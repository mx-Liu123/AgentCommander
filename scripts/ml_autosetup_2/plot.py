import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np

def draw_plots(X_noisy, y_true, y_pred, output_dir, params, X_clean=None):
    """
    Draws a dashboard with:
    1. Phase Error Distribution (Histogram)
    2. Input Signal Example (Noisy vs Clean)
    3. Phase reconstruction plots (Unwrapped phase)
    
    Args:
        X_noisy: (N, P, L) - Input passed to model
        y_true:  (N, L, 2) - Ground truth (cos, sin)
        y_pred:  (N, L, 2) - Prediction (cos, sin)
        output_dir: Path to save plot
        params: Config dict
        X_clean: (N, P, L) - Optional clean signal for comparison
    """
    filename = "best_result.png"
    filepath = os.path.join(output_dir, filename)
    
    num_samples, seq_len, _ = y_true.shape
    
    # 1. Convert (cos, sin) to Unwrapped Phase
    # y shape is (N, L, 2) -> (cos, sin)
    phi_true = np.arctan2(y_true[..., 1], y_true[..., 0]) # (N, L)
    phi_pred = np.arctan2(y_pred[..., 1], y_pred[..., 0]) # (N, L)
    
    # Unwrap along time axis
    phi_true_u = np.unwrap(phi_true, axis=1)
    phi_pred_u = np.unwrap(phi_pred, axis=1)
    
    # 2. Calculate Metrics (MAE per sample in degrees) -- USING WRAPPED ERROR
    # Circular difference: shortest distance on the circle
    diff = phi_true - phi_pred
    diff_wrapped = (diff + np.pi) % (2 * np.pi) - np.pi
    sample_mae = np.mean(np.abs(diff_wrapped), axis=1) # (N,)
    
    # --- Select Indices ---
    # 3 Worst (High MAE)
    num_worst = min(3, num_samples)
    worst_indices = np.argsort(sample_mae)[-num_worst:][::-1]
    
    # 1 Best (Low MAE)
    best_indices = np.argsort(sample_mae)[:1]
    
    # 2 Random (excluding best/worst)
    exclude = np.concatenate([worst_indices, best_indices])
    candidates = np.setdiff1d(np.arange(num_samples), exclude)
    num_random = min(2, len(candidates))
    random_indices = np.random.choice(candidates, size=num_random, replace=False)
    
    indices = np.concatenate([best_indices, random_indices, worst_indices])
    labels = ['Best'] * len(best_indices) + ['Random'] * len(random_indices) + ['Worst'] * len(worst_indices)

    # --- Plotting ---
    num_sample_rows = len(indices)
    # Adjust figure height based on number of plots
    fig = plt.figure(figsize=(14, 5 + 3 * num_sample_rows))
    gs = gridspec.GridSpec(num_sample_rows + 1, 2, height_ratios=[1.2] + [1] * num_sample_rows)
    
    # A. Error Histogram
    ax_hist = fig.add_subplot(gs[0, 0])
    mae_deg = np.degrees(sample_mae)
    # Clip extreme outliers for nicer histogram
    limit = np.percentile(mae_deg, 98)
    ax_hist.hist(mae_deg, bins=50, range=(0, limit), density=True, color='skyblue', edgecolor='white', alpha=0.7)
    ax_hist.set_title("Phase MAE Distribution (Degrees, Wrapped)")
    ax_hist.set_xlabel("Mean Absolute Error (°)")
    ax_hist.axvline(np.mean(mae_deg), color='red', linestyle='--', label=f'Mean: {np.mean(mae_deg):.2f}°')
    ax_hist.axvline(np.median(mae_deg), color='orange', linestyle=':', label=f'Median: {np.median(mae_deg):.2f}°')
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)
    
    # B. Input Signal Example
    ax_sig = fig.add_subplot(gs[0, 1])
    # Show the worst sample to see why it failed
    idx_example = worst_indices[0] 
    t = np.arange(seq_len)
    
    # Plot Pulsar 0 (first channel)
    ax_sig.plot(t, X_noisy[idx_example, 0, :], color='gray', alpha=0.6, lw=1, label='Noisy Input (P0)')
    if X_clean is not None:
        ax_sig.plot(t, X_clean[idx_example, 0, :], color='blue', alpha=0.8, lw=1.5, label='Clean Signal (P0)')
        
    ax_sig.set_title(f"Input Signal Example (Worst Sample {idx_example}, Wrapped MAE)")
    ax_sig.legend(loc='upper right')
    ax_sig.grid(alpha=0.3)
    
    # C. Sample Phase Plots
    for i, (idx, label) in enumerate(zip(indices, labels)):
        ax = fig.add_subplot(gs[i+1, :])
        
        ax.plot(t, phi_true_u[idx], color='black', lw=2.5, alpha=0.8, label='Ground Truth (Unwrapped)')
        ax.plot(t, phi_pred_u[idx], color='cyan', lw=2, ls='--', alpha=0.9, label='Prediction (Unwrapped)')
        
        ax.set_title(f"{label} Sample {idx}: Phase Reconstruction (Wrapped MAE: {mae_deg[idx]:.2f}°)")
        ax.set_ylabel("Phase (rad)")
        ax.set_xlim(0, seq_len)
        if i == 0: ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    try:
        plt.savefig(filepath, dpi=150)
        print(f"Plot saved to {filepath}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()
    return [filename]