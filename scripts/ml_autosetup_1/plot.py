import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np

def downsample_timeseries(X_single, new_len: int) -> np.ndarray:
    """Helper to downsample a single 1D array."""
    old_len = len(X_single)
    idx = np.linspace(0, old_len - 1, new_len).astype(int)
    return X_single[idx]

def draw_plots(X_test, y_test, y_pred, output_dir, params):
    """
    Draws a dashboard with:
    1. Phase Error Distribution (Histogram)
    2. Input Signal Example (Noisy vs Clean matched by index)
    3. 3 Worst samples comparing Ground Truth Phase vs Predicted Phase.
    """
    filename = "best_result.png"
    filepath = os.path.join(output_dir, filename)
    
    num_samples = len(X_test)
    epsilon = 1e-8
    y_true_flat = y_test.reshape(num_samples, -1)
    y_pred_flat = y_pred.reshape(num_samples, -1)

    error_energy = np.sum((y_true_flat - y_pred_flat) ** 2, axis=1)
    true_energy = np.sum(y_true_flat ** 2, axis=1)
    sample_nmses = error_energy / (true_energy + epsilon)
    
    k = min(3, num_samples)
    worst_indices = np.argsort(sample_nmses)[-k:][::-1]
    indices = worst_indices
    
    # Save worst samples
    np.save(os.path.join(output_dir, 'worst_samples.npy'), {'X': X_test[indices], 'Y': y_test[indices]})

    # --- Match Clean Signal Logic ---
    clean_sig_example = None
    naive_ds_sig = None
    try:
        DATA_DIR = "/home/liumx/test_data"
        X_NOISY_ALL = np.load(os.path.join(DATA_DIR, "X_dev.npy"))
        X_RAW_ALL = np.load(os.path.join(DATA_DIR, "X_dev_raw.npy"))
        
        # Match the first 'worst' sample to find its original index
        sample_to_match = X_test[indices[0]]
        matches = np.where(np.all(X_NOISY_ALL == sample_to_match, axis=1))[0]
        
        if len(matches) > 0:
            orig_idx = matches[0]
            raw_data = X_RAW_ALL[orig_idx]
            if raw_data.ndim > 1: raw_data = raw_data.ravel()
            
            # 1. Reconstruct Clean Signal (Centered + Downsample)
            raw_centered = raw_data - np.mean(raw_data)
            clean_sig_example = downsample_timeseries(raw_centered, 400)

            # 2. Reconstruct Naive Downsampled Signal (Noisy 1000 -> Naive 400)
            # Must replicate noise generation exactly
            np.random.seed(42)
            # Re-generate full noise params to match sequence
            snrs = np.random.choice(np.arange(20, 30), size=len(X_RAW_ALL))
            signal_norm = np.linalg.norm(X_RAW_ALL.reshape(len(X_RAW_ALL), -1), axis=1)
            noise_std = (signal_norm / snrs).reshape(len(X_RAW_ALL), *([1] * (X_RAW_ALL.ndim - 1)))
            
            # Generate noise just for this sample? No, RNG state is global.
            # We must generate the full noise matrix to get the correct values for orig_idx.
            # Using float32 to save memory if needed, but standard is float64.
            # Generating 95k * 1k doubles is ~760MB. Acceptable.
            noise_matrix = np.random.normal(0.0, 1.0, size=X_RAW_ALL.shape)
            full_noise = noise_matrix * noise_std
            
            noisy_1000_centered = raw_centered + full_noise[orig_idx].ravel()
            naive_ds_sig = downsample_timeseries(noisy_1000_centered, 400)
            
    except Exception as e:
        print(f"Note: Could not load/match clean signal: {e}")

    # Setup Figure
    fig = plt.figure(figsize=(12, 14))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])
    
    # 1. Histogram
    ax_hist = fig.add_subplot(gs[0, 0])
    mae_deg = np.degrees(np.mean(np.abs(y_true_flat - y_pred_flat), axis=1))
    limit = np.percentile(mae_deg, 95)
    ax_hist.hist(mae_deg, bins=50, range=(0, limit), density=True, alpha=0.6, color='skyblue', edgecolor='white')
    ax_hist.axvline(np.mean(mae_deg), color='red', linestyle='--', label=f'mean={np.mean(mae_deg):.2f}°')
    ax_hist.axvline(np.median(mae_deg), color='orange', linestyle=':', label=f'median={np.median(mae_deg):.2f}°')
    ax_hist.set_title("Phase alignment over validation set")
    ax_hist.set_xlim(0, limit)
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    # 2. Input Signal Example (Noisy vs Clean)
    ax_sig = fig.add_subplot(gs[0, 1])
    noisy_data = X_test[indices[0]].ravel()
    t = np.arange(len(noisy_data))
    
    # Plot Resampled (Input)
    ax_sig.scatter(t, noisy_data, s=10, color='gray', alpha=0.6, label='Resampled Input (Model Input)')
    
    # Plot Naive Downsampled (Comparison)
    if naive_ds_sig is not None:
        ax_sig.plot(t, naive_ds_sig, color='orange', linewidth=1.0, alpha=0.5, label='Naive Downsample (No Smoothing)')

    # Plot Clean
    if clean_sig_example is not None:
        ax_sig.plot(t, clean_sig_example, color='blue', linewidth=1.5, alpha=0.9, label='Clean (Centered)')
    
    ax_sig.set_title(f"Input Signal Comparison (Worst Sample {indices[0]})")
    ax_sig.legend(fontsize='small')
    ax_sig.grid(True, alpha=0.3)

    # 3-5. Worst Samples
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(gs[i+1, :])
        ax.plot(y_test[idx].ravel(), label="Ground Truth", color='black', alpha=0.8)
        ax.plot(y_pred[idx].ravel(), label="Prediction", color='cyan', linestyle='--', alpha=0.9)
        ax.set_title(f"Sample {idx}: Phase Reconstruction (NMSE: {sample_nmses[idx]:.4f})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Validation Results & Worst Cases Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return [filename]
