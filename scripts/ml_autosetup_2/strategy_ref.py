import os, sys, math, time, random, platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# 1. STRATEGY CONFIGURATION (Optimizable)
# ==========================================================
# Training Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters (Agent can modify these freely)
HPARAMS = {
    "d_model": 128,
    "depth": 6,
    "heads": 4,
    "d_ff": 512,
    "ds_stride": 2,
    "p_drop": 0.05,
    "lr": 8e-4,
    "batch_size": 128,
    "epochs": 10
}

# ==========================================================
# 2. DATA LOADING (Protocol - DO NOT MODIFY)
# ==========================================================
from experiment_setup import load_and_split_data

# ==========================================================
# 3. MODEL DEFINITION (Agent can redefine freely)
# ==========================================================
class PhaseTransformerRealisationFast(nn.Module):
    def __init__(self, P, L, d_model, depth, heads, d_ff, p_drop, ds_stride):
        super().__init__()
        # Simple Transformer-like structure for reference
        self.embedding = nn.Linear(L, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, heads, d_ff, p_drop, batch_first=True),
            num_layers=depth
        )
        self.fc_out = nn.Linear(d_model, L * 2) # Output cos/sin
        self.P = P
        self.L = L

    def forward(self, x):
        # x: (B, P, L) -> flatten -> (B, P*L) ? No, let's treat as sequence of length P?
        # This is just a dummy reference implementation
        B, P, L = x.shape
        x_emb = self.embedding(x) # (B, P, d_model)
        x_enc = self.encoder(x_emb)
        out = self.fc_out(x_enc) # (B, P, L*2)
        # Reshape to (B, L, 2) ? Usually output is phase map.
        # Let's assume output is (B, L, 2) unit vector
        return out.mean(dim=1).view(B, L, 2)

# ==========================================================
# 4. EXPORT INTERFACE (Required for Evaluator)
# ==========================================================
def load_trained_model(path, device):
    """
    Factory function for Evaluator to load the model without knowing class details.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
        
    checkpoint = torch.load(path, map_location=device)
    
    # Load Config from checkpoint (Best Practice for Evolution)
    # If not present, fallback to current global HPARAMS
    config = checkpoint.get('config', HPARAMS) 
    
    # Instantiate Model with saved config
    # Note: P and L usually come from data, but here we might need to know them.
    # We can save them in config too, or assume fixed.
    P = config.get('P', 2)
    L = config.get('L', 1000)
    
    model = PhaseTransformerRealisationFast(
        P, L, 
        d_model=config.get('d_model', 128),
        depth=config.get('depth', 6),
        heads=config.get('heads', 4),
        d_ff=config.get('d_ff', 512),
        p_drop=config.get('p_drop', 0.05),
        ds_stride=config.get('ds_stride', 2)
    )
    
    # Load Weights
    sd = checkpoint['model_state_dict']
    # Handle torch.compile prefix if present
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    
    return model

# ==========================================================
# 5. TRAINING EXECUTION
# ==========================================================
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    X_tr, X_va, phi_tr, phi_va = load_and_split_data()
    P, R, L = X_tr.shape
    
    # Update HPARAMS with data shape
    HPARAMS['P'] = P
    HPARAMS['L'] = L
    
    # Instantiate
    model = PhaseTransformerRealisationFast(
        P, L, HPARAMS['d_model'], HPARAMS['depth'], HPARAMS['heads'], 
        HPARAMS['d_ff'], HPARAMS['p_drop'], HPARAMS['ds_stride']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=HPARAMS['lr'])
    criterion = nn.MSELoss()
    
    print("Starting training...")
    # Dummy Training Loop
    model.train()
    for epoch in range(1, 3): # Just 2 epochs for ref check
        # ... (Real training logic goes here) ...
        print(f"Epoch {epoch} complete.")
        
    # Save Model + Config
    save_path = "best_fast.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": HPARAMS, # Save current config!
        "mu_x": torch.zeros(1), # Dummy stats
        "std_x": torch.ones(1)
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    run_training()