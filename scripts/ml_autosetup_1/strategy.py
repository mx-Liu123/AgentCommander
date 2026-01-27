import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time as time_lib

def get_search_configs():
    return [{
        "lr": 0.001,
        "epochs": 100,
        "batch_size": 256,
        "hidden_size": 128, 
        "cnn_channels": 64,  
        "aux_weight": 5.0,    # Sobolev (Huber)
        "wave_weight": 0.5,
        "trend_weight": 5.0,  # Global slope magnitude
        "sign_weight": 12.0,  # Directional penalty (Democratic voting)
        "anc_weight": 3.0     # Phasor anchor
    }]

class Strategy:
    def __init__(self, params=None):
        self.params = params if params else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.batch_size = self.params.get("batch_size", 256)

    def fit(self, X, y, deadline=None):
        torch.manual_seed(42)
        if torch.cuda.is_available(): torch.cuda.manual_seed(42)

        # Target and derivative preparation
        y_use = y.squeeze(-1) if (y.ndim == 3 and y.shape[-1] == 1) else y
        y_grad = np.gradient(y_use, axis=1)
        
        # Per-sample normalization of input X
        X_mean = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True) + 1e-15
        X_norm = (X - X_mean) / X_std
        
        X_tensor = torch.from_numpy(X_norm).float()
        if X_tensor.ndim == 2: X_tensor = X_tensor.unsqueeze(-1)
        
        y_multi = np.stack([y_use, y_grad], axis=-1)
        dataset = TensorDataset(X_tensor, torch.from_numpy(y_multi).float())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        class RobustS_PINN_v11(nn.Module):
            def __init__(self, in_dim, h_size, c_chan):
                super().__init__()
                # Dilated CNN for temporal feature extraction
                self.cnn = nn.Sequential(
                    nn.Conv1d(in_dim, c_chan, 7, padding=3),
                    nn.GroupNorm(4, c_chan), nn.ReLU(),
                    nn.Conv1d(c_chan, c_chan, 7, padding=6, dilation=2),
                    nn.GroupNorm(4, c_chan), nn.ReLU(),
                    nn.Conv1d(c_chan, c_chan, 7, padding=12, dilation=4),
                    nn.GroupNorm(4, c_chan), nn.ReLU()
                )
                self.rnn = nn.GRU(c_chan, h_size, num_layers=2, batch_first=True, bidirectional=True)
                
                # Voting Head: Democratic Sign Consensus (Robust Direction)
                self.sign_vote = nn.Sequential(
                    nn.Linear(h_size * 2, h_size // 2),
                    nn.ReLU(),
                    nn.Linear(h_size // 2, 1)
                )
                
                # Local magnitude of frequency
                self.freq_fc = nn.Linear(h_size * 2, 1)
                
                # Anchor Phasor predictor [cos y0, sin y0]
                self.anc_fc = nn.Sequential(
                    nn.Linear(h_size * 4, h_size),
                    nn.ReLU(),
                    nn.Linear(h_size, 2)
                )
                
                # Local residual correction
                self.res_fc = nn.Linear(h_size * 2, 1)

            def forward(self, x):
                B, T, _ = x.shape
                x = x.permute(0, 2, 1)
                x = self.cnn(x)
                x = x.permute(0, 2, 1)
                r_out, _ = self.rnn(x) # Shape: (B, T, h_size * 2)
                
                # Aggregate global context for anchor
                r_mean = torch.mean(r_out, dim=1)
                r_max, _ = torch.max(r_out, dim=1)
                g_feat = torch.cat([r_mean, r_max], dim=1) # (B, h_size * 4)
                
                # 1. Sign Voting consensus
                v_logits = self.sign_vote(r_out) # (B, T, 1)
                direction = torch.tanh(torch.mean(v_logits, dim=1)).view(B, 1, 1)
                
                # 2. Frequency Magnitude
                freq_mag = F.softplus(self.freq_fc(r_out))
                freq = freq_mag * direction
                
                # 3. Anchor Phasor [cos y0, sin y0]
                anc_raw = self.anc_fc(g_feat)
                anc_norm = F.normalize(anc_raw, dim=1, eps=1e-6)
                y0 = torch.atan2(anc_norm[:, 1], anc_norm[:, 0]).view(B, 1, 1)
                
                # 4. Integrate Trajectory
                phase_int = torch.cumsum(freq, dim=1) - freq[:, 0:1, :]
                phase_res = torch.tanh(self.res_fc(r_out)) * 2.0
                
                return y0 + phase_int + phase_res, freq, anc_norm, v_logits

        self.model = RobustS_PINN_v11(X_tensor.shape[2], self.params["hidden_size"], self.params["cnn_channels"]).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["lr"])
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.params["lr"]*2, 
                                                 steps_per_epoch=len(dataloader), epochs=self.params["epochs"])

        self.model.train()
        print(f"[Robust S-PINN v11] Voting Sign Consensus | Phasor Anchor", flush=True)
        for epoch in range(self.params["epochs"]):
            if deadline and time_lib.time() > deadline - 30: break
            
            total_nmse = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                p_out, f_out, a_out, v_out = self.model(batch_X)
                
                # --- Loss 1: NMSE ---
                error = (p_out[:, :, 0] - batch_y[:, :, 0])**2
                energy = torch.sum(batch_y[:, :, 0]**2, dim=1, keepdim=True) + 1e-8
                loss_nmse = torch.mean(torch.sum(error, dim=1, keepdim=True) / energy)
                
                # --- Loss 2: Sobolev (Velocity) Huber ---
                loss_f = F.smooth_l1_loss(f_out[:, :, 0], batch_y[:, :, 1])
                
                # --- Loss 3: Global Trend Magnitude Huber ---
                gt_mean_v = torch.mean(batch_y[:, :, 1], dim=1)
                pred_mean_v = torch.mean(f_out[:, :, 0], dim=1)
                loss_t = F.smooth_l1_loss(pred_mean_v, gt_mean_v)
                
                # --- Loss 4: Sign/Direction Voting Consistency ---
                gt_sign = torch.sign(gt_mean_v).view(-1, 1, 1)
                loss_s = torch.mean(F.relu(-v_out * gt_sign))
                
                # --- Loss 5: Anchor Phasor Vector MSE ---
                gt_anc_vec = torch.stack([torch.cos(batch_y[:, 0, 0]), torch.sin(batch_y[:, 0, 0])], dim=1)
                loss_a = F.mse_loss(a_out, gt_anc_vec)
                
                # --- Loss 6: Periodic Wave Loss ---
                loss_w = torch.mean(1 - torch.cos(p_out[:, :, 0] - batch_y[:, :, 0]))
                
                loss = (loss_nmse + 
                        self.params["aux_weight"] * loss_f + 
                        self.params["trend_weight"] * loss_t + 
                        self.params["sign_weight"] * loss_s + 
                        self.params["anc_weight"] * loss_a + 
                        self.params["wave_weight"] * loss_w)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_nmse += loss_nmse.item()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d} | Train NMSE: {total_nmse/len(dataloader):.4f}", flush=True)

    def predict(self, X):
        if self.model is None: return np.zeros_like(X)
        self.model.eval()
        X_mean, X_std = np.mean(X, axis=1, keepdims=True), np.std(X, axis=1, keepdims=True) + 1e-15
        X_tensor = torch.from_numpy((X - X_mean) / X_std).float()
        if X_tensor.ndim == 2: X_tensor = X_tensor.unsqueeze(-1)
        
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                p_out, _, _, _ = self.model(X_tensor[i:i+self.batch_size].to(self.device))
                preds.append(p_out[:, :, 0].cpu().numpy())
        
        res = np.concatenate(preds, axis=0)
        return res[..., np.newaxis] if X.ndim == 3 else res
