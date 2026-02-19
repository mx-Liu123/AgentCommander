import torch
import torch.nn.functional as F
import math
import numpy as np

# --- Interface for Evaluator ---
HIGHER_IS_BETTER = False

def calculate_score(y_true, y_pred):
    """
    Adapter function for the evaluator.
    y_true: Ground truth phase values (numpy array)
    y_pred: Predicted phase values (numpy array)
    """
    return phase_mean_absolute_error(y_pred, y_true)

# --- Core Logic ---

def _normalize_unit(y, eps=1e-7):
    """Normalize a tensor of vectors to unit length."""
    norm = torch.linalg.norm(y, dim=-1, keepdim=True)
    return y / torch.clamp(norm, min=eps)

def get_loss_fns(kappa, l_smooth, l_spect, device):
    """
    Returns the total loss function composed of its parts.
    Hyperparameters are passed explicitly.
    """
    def von_mises_nll(y_pred, y_true):
        # y_pred, y_true shape: (B, L, 2)
        yp, yt = _normalize_unit(y_pred), _normalize_unit(y_true)
        # Dot product along the last dimension (cos(angle_diff))
        dot_prod = torch.sum(yp * yt, dim=-1)
        return torch.mean(-kappa * dot_prod)

    def circular_smoothness_loss(y):
        y = _normalize_unit(y)
        # atan2 returns values in (-pi, pi]
        a = torch.atan2(y[..., 1], y[..., 0])
        # Calculate circular difference: (a[i+1] - a[i] + pi) % 2pi - pi
        d = (a[:, 1:] - a[:, :-1] + math.pi) % (2 * math.pi) - math.pi
        return torch.mean(d**2)

    def spectral_unit_loss(y_pred, y_true):
        yp, yt = _normalize_unit(y_pred), _normalize_unit(y_true)
        zp = torch.complex(yp[..., 0], yp[..., 1]) # (B, L)
        zt = torch.complex(yt[..., 0], yt[..., 1]) # (B, L)
        
        # FFT over the last dimension (L)
        # Using fft without 'n' defaults to input length, which is correct
        Pp = torch.fft.fft(zp) 
        Pt = torch.fft.fft(zt)
        
        Pm, Tm = Pp.abs(), Pt.abs()
        
        # Normalize the spectrum amplitude to unit norm
        # This makes the loss invariant to signal energy scaling
        Pm = Pm / (Pm.norm(dim=-1, keepdim=True) + 1e-8)
        Tm = Tm / (Tm.norm(dim=-1, keepdim=True) + 1e-8)
        
        return F.mse_loss(Pm, Tm)

    def total_loss(y_pred, y_true):
        lv = von_mises_nll(y_pred, y_true)
        ls = circular_smoothness_loss(y_pred)
        lp = spectral_unit_loss(y_pred, y_true)
        
        tot = lv + l_smooth * ls + l_spect * lp
        parts = {"total": tot.item(), "von_mises": lv.item(), "smooth": ls.item(), "spectral": lp.item()}
        return tot, parts
        
    return total_loss

def phase_mean_absolute_error(phi_pred, phi_true):
    """
    Calculates the Mean Absolute Error for phase, handling angle wrapping.
    phi_pred and phi_true are numpy arrays in radians.
    """
    if isinstance(phi_pred, torch.Tensor):
        phi_pred = phi_pred.cpu().numpy()
    if isinstance(phi_true, torch.Tensor):
        phi_true = phi_true.cpu().numpy()

    # Calculate difference in complex plane to handle wrapping
    # exp(i*pred) / exp(i*true) = exp(i*(pred-true))
    # angle(exp(i*(pred-true))) gives the wrapped difference in (-pi, pi]
    diff = np.angle(np.exp(1j * (phi_pred - phi_true)))
    
    return np.mean(np.abs(diff))