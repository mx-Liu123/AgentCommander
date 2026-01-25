import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys

def get_search_configs():
    """
    Returns optimized configuration for KGF-v10 (Steady-Trend Kinematic Fusion).
    Combines stable 1st-order integration with pyramid features and wave-matching.
    """
    return [
        {
            "lr": 0.001,
            "epochs": 20,
            "batch_size": 64,
            "hidden_size": 128, 
            "cnn_channels": 64,  
            "aux_weight": 0.8,      # Frequency focus
            "wave_weight": 25.0,    # Wave matching focus
            "smooth_weight": 5.0    # TV regularization
        }
    ]

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.size()
        y = torch.mean(x, dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y

class PhysicalCNN(nn.Module):
    def __init__(self, input_dim, channels):
        super(PhysicalCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(channels)
        
        # Scale 1: Short (7)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=15, padding=7, groups=channels)
        self.bn2 = nn.BatchNorm1d(channels)
        
        # Scale 2: Medium (31) with Dilation 2
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=31, padding=30, groups=channels, dilation=2)
        self.bn3 = nn.BatchNorm1d(channels)
        
        # Pyramid fusion bottleneck
        self.bottleneck = nn.Conv1d(channels * 3, channels, kernel_size=1)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1))) + x1
        x3 = self.relu(self.bn3(self.conv3(x2))) + x2
        
        # Pyramid: Concatenate multi-scale features
        x_pyr = torch.cat([x1, x2, x3], dim=1)
        x = self.relu(self.bottleneck(x_pyr))
        x = self.se(x)
        return x

class PhysicsInformedModel(nn.Module):
    def __init__(self, input_dim, hidden_size, cnn_channels):
        super(PhysicsInformedModel, self).__init__()
        self.cnn = PhysicalCNN(input_dim, cnn_channels)
        self.rnn = nn.GRU(input_size=cnn_channels, hidden_size=hidden_size, 
                          num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size * 2, 2) # [freq_logit, res_logit]
        self.anchor_fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        rnn_out, _ = self.rnn(x)
        
        raw_out = self.fc(rnn_out)
        freq = F.softplus(raw_out[:, :, 0:1])
        
        # Global Anchor (y0) to fix the integration constant
        y0 = self.anchor_fc(torch.mean(rnn_out, dim=1))
        
        # Structural Integration (Trend path)
        phase_int = torch.cumsum(freq, dim=1) - freq[:, 0:1, :]
        
        # Residual Correction path (Bounded to prevent non-physical jumps)
        phase_res = torch.tanh(raw_out[:, :, 1:2]) * 3.0
        
        phase_final = y0.unsqueeze(1) + phase_int + phase_res
        
        return torch.cat([phase_final, freq], dim=-1)

class Strategy:
    def __init__(self, params=None):
        self.params = params if params else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        self.lr = self.params.get("lr", 0.001)
        self.epochs = self.params.get("epochs", 20)
        self.batch_size = self.params.get("batch_size", 64)
        self.hidden_size = self.params.get("hidden_size", 128)
        self.cnn_channels = self.params.get("cnn_channels", 64)
        self.aux_weight = self.params.get("aux_weight", 0.8)
        self.wave_weight = self.params.get("wave_weight", 25.0)
        self.smooth_weight = self.params.get("smooth_weight", 5.0)
        
        self.x_mean = 0.0
        self.x_std = 1.0

    def fit(self, X, y):
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.empty_cache()

        y_use = y.squeeze(-1) if (y.ndim == 3 and y.shape[-1] == 1) else y
        # Target for instantaneous frequency (Sobolev derivative)
        y_grad = np.gradient(y_use, axis=1) 
        
        print(f"[Physics] Steady-Trend KGF Training | SeqLen: {y_use.shape[1]} | AvgFreq: {np.mean(np.abs(y_grad)):.4e}", flush=True)

        self.x_mean, self.x_std = np.mean(X), np.std(X) + 1e-15
        X_norm = (X - self.x_mean) / self.x_std
        X_tensor = torch.from_numpy(X_norm).float()
        if X_tensor.ndim == 2: X_tensor = X_tensor.unsqueeze(-1)
        
        y_multi = np.stack([y_use, y_grad], axis=-1)
        dataset = TensorDataset(X_tensor, torch.from_numpy(y_multi).float())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = PhysicsInformedModel(
            input_dim=X_tensor.shape[2],
            hidden_size=self.hidden_size,
            cnn_channels=self.cnn_channels
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            e_pha, e_wav = 0.0, 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(batch_X) 
                
                # 1. Trend Phase MSE
                loss_phase = criterion(pred[:, :, 0], batch_y[:, :, 0])
                # 2. Sobolev Frequency MSE
                loss_freq = criterion(pred[:, :, 1], batch_y[:, :, 1])
                # 3. Wave Matching (Periodic alignment)
                loss_wave = torch.mean(1 - torch.cos(pred[:, :, 0] - batch_y[:, :, 0]))
                # 4. Smoothness TV
                loss_tv = torch.mean((pred[:, 1:, 1] - pred[:, :-1, 1])**2)
                
                loss = loss_phase + self.aux_weight * loss_freq + self.wave_weight * loss_wave + self.smooth_weight * loss_tv
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                e_pha += loss_phase.item()
                e_wav += loss_wave.item()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d} | Pha: {e_pha/len(dataloader):.4e} | Wav: {e_wav/len(dataloader):.4e}", flush=True)

    def predict(self, X):
        if self.model is None: return np.zeros_like(X)
        self.model.eval()
        X_norm = (X - self.x_mean) / self.x_std
        X_tensor = torch.from_numpy(X_norm).float()
        if X_tensor.ndim == 2: X_tensor = X_tensor.unsqueeze(-1)
        
        dataloader = DataLoader(TensorDataset(X_tensor), batch_size=128, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                out = self.model(batch[0].to(self.device))
                preds.append(out[:, :, 0].cpu().numpy())
        
        res = np.concatenate(preds, axis=0)
        return res[..., np.newaxis] if X.ndim == 3 else res