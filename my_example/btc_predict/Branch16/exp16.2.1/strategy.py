import os
import math
import numpy as np
import pandas as pd
import talib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Import the fixed model definitions
from model import CnnLstmNetwork, StudentT_NLLLoss, load_or_train_model

# Set deterministic behavior for reproducibility
seed = int(os.getenv("SEED", "42"))
torch.manual_seed(seed)
np.random.seed(seed)

def rolling_hurst(series, window=100):
    """
    Standard Rescaled Range (R/S) Hurst exponent on log-returns.
    H < 0.5: Mean Reverting
    H = 0.5: Random Walk
    H > 0.5: Persistent/Trending
    """
    def calc_hurst(x):
        if len(x) < 50: return 0.5
        z = np.diff(np.log(x + 1e-8))
        if len(z) < 20: return 0.5
        mean_adj = z - np.mean(z)
        cum_sum = np.cumsum(mean_adj)
        r = np.max(cum_sum) - np.min(cum_sum)
        s = np.std(z) + 1e-8
        return np.log(r / s) / np.log(len(z))
    
    return series.rolling(window).apply(calc_hurst).fillna(0.5)

def get_search_configs():
    """
    Search space for Regime-Adaptive Precision (RAP) Strategy.
    Restoring conservative exit gates while maintaining dynamic sigma.
    """
    base = {
        "required_train_days": 150,
        "learning_rate": 0.001,
        "hidden_dim": 128,
        "seq_len": 80, 
        "dropout_rate": 0.4,
        "initial_epochs": 200, 
        "incremental_epochs": 50,
        "batch_size": 256
    }
    
    configs = []
    
    # 1. Balanced RAP (Winning Parameters from Parent)
    c1 = base.copy()
    c1.update({
        "sigma_thresh": 620, "adx_thresh": 12, "atr_mult_base": 2.6, 
        "rsi_upper": 75, "rsi_lower": 25, "t_remaining_min": 30, 
        "er_thresh": 0.14, "tension_limit": 0.38, "divergence_thresh": 0.04,
        "use_breadth": True, "breadth_gate": 0.25, "vti_limit": 0.07,
        "hurst_thresh": 0.53, "pressure_trigger": 1.10, "obv_corr_gate": 0.30,
        "ignition_thr": 1.30,
        "tension_gate": 0.05, "tension_boost": 0.10,
        "exit_tension_gate": 0.075,
        "adx_sigma_slope": 0.005,
        "accum_bypass": True
    })
    configs.append(c1)

    # 2. More Conservative (Lower Sigma, Higher Hurst)
    c2 = c1.copy()
    c2.update({
        "sigma_thresh": 600,
        "hurst_thresh": 0.55
    })
    configs.append(c2)

    # 3. Aggressive Trending (Higher Sigma, Lower Exit Gate)
    c3 = c1.copy()
    c3.update({
        "sigma_thresh": 640,
        "exit_tension_gate": 0.070
    })
    configs.append(c3)

    return configs

class ModelStrategy:
    def __init__(self, params=None):
        self.model = None
        self.scaler = StandardScaler()
        self.params = params if params else get_search_configs()[0]
        
        self.required_train_days = int(self.params.get("required_train_days", 150))
        self.seq_len = int(self.params.get("seq_len", 80))
        self.batch_size = int(self.params.get("batch_size", 256))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = float(self.params.get("learning_rate", 0.001))
        self.hidden_dim = int(self.params.get("hidden_dim", 128))
        self.dropout_rate = float(self.params.get("dropout_rate", 0.4))
        
        # Strategy Thresholds
        self.sigma_thresh = float(self.params.get("sigma_thresh", 615))
        self.adx_thresh = float(self.params.get("adx_thresh", 14))
        self.atr_mult_base = float(self.params.get("atr_mult_base", 2.5))
        self.rsi_upper = float(self.params.get("rsi_upper", 72))
        self.rsi_lower = float(self.params.get("rsi_lower", 28))
        self.t_remaining_min = float(self.params.get("t_remaining_min", 35))
        self.er_thresh = float(self.params.get("er_thresh", 0.16))
        self.tension_limit = float(self.params.get("tension_limit", 0.35))
        self.divergence_thresh = float(self.params.get("divergence_thresh", 0.03))
        
        self.use_breadth = self.params.get("use_breadth", True)
        self.breadth_gate = float(self.params.get("breadth_gate", 0.3))
        self.vti_limit = float(self.params.get("vti_limit", 0.06))
        
        # New Params
        self.hurst_thresh = float(self.params.get("hurst_thresh", 0.51))
        self.pressure_trigger = float(self.params.get("pressure_trigger", 1.15))
        self.obv_corr_gate = float(self.params.get("obv_corr_gate", 0.35))
        self.ignition_thr = float(self.params.get("ignition_thr", 1.35))
        
        self.tension_gate = float(self.params.get("tension_gate", 0.05))
        self.tension_boost = float(self.params.get("tension_boost", 0.15))
        
        # New Params for Exit
        self.exit_tension_gate = float(self.params.get("exit_tension_gate", 0.05))
        
        # RAP Params
        self.adx_sigma_slope = float(self.params.get("adx_sigma_slope", 0.0))
        self.accum_bypass = self.params.get("accum_bypass", False)
        
        self.request_data = None 

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from evaluator import log_move_adaptive_extrema, DETECTOR_CONFIG
        X = df.copy()
        f = {}
        
        close, high, low, volume = X['close'].values.astype(float), X['high'].values.astype(float), X['low'].values.astype(float), X['volume'].values.astype(float)
        
        f['ret_1'] = pd.Series(close).pct_change(1).fillna(0).values
        f['ret_10'] = pd.Series(close).pct_change(10).fillna(0).values
        f['log_vol'] = np.log1p(volume)
        
        # Technicals
        f['rsi_14'] = talib.RSI(close, 14) / 100.0
        f['mfi_14'] = talib.MFI(high, low, close, volume, 14) / 100.0
        f['atr_14'] = talib.ATR(high, low, close, 14) / (close + 1e-8)
        
        # Fractal Features
        f['hurst_100'] = rolling_hurst(pd.Series(close), window=100).values
        f['hurst_grad'] = pd.Series(f['hurst_100']).diff(5).fillna(0).values
        
        # Volume-Price Features
        obv = talib.OBV(close, volume)
        f['obv_corr'] = pd.Series(obv).rolling(20).corr(pd.Series(close)).fillna(0).values
        
        bp = close - low
        sp = high - close
        f['pressure_ratio'] = (pd.Series(bp).rolling(5).mean() / (pd.Series(sp).rolling(5).mean() + 1e-8)).clip(0, 5).fillna(1.0).values
        
        atr_20, atr_100 = talib.ATR(high, low, close, 20), talib.ATR(high, low, close, 100)
        f['vol_regime'] = (atr_20 / (atr_100 + 1e-8)).clip(0, 5)
        
        v_ma = pd.Series(volume).rolling(50).mean()
        v_std = pd.Series(volume).rolling(50).std()
        f['vol_intensity'] = ((pd.Series(volume) - v_ma) / (v_std + 1e-8)).fillna(0).values
        
        f['adx_14'] = talib.ADX(high, low, close, 14) / 100.0
        f['adx_grad'] = (pd.Series(f['adx_14']).diff(3).fillna(0)).values
        
        # Kaufman Efficiency Ratio (ER)
        net_chg = pd.Series(close).diff(10).abs()
        abs_chg_sum = pd.Series(close).diff(1).abs().rolling(10).sum()
        f['er_10'] = (net_chg / (abs_chg_sum + 1e-8)).fillna(0).values
        f['rsi_vel'] = (pd.Series(f['rsi_14']).diff(3).fillna(0)).values
        
        ema20, ema50 = talib.EMA(close, 20), talib.EMA(close, 50)
        f['ema_spread'] = (ema20 - ema50) / (close + 1e-8)
        f['ema_slope_20'] = (pd.Series(ema20).diff(3) / (close + 1e-8)).fillna(0).values
        
        rsi_roc = pd.Series(f['rsi_14']).diff(5)
        price_roc = pd.Series(close).pct_change(5)
        f['mom_divergence'] = (rsi_roc - price_roc).fillna(0).values
        
        # Vol-Trend Interaction (VTI)
        v_ratio = pd.Series(volume) / (pd.Series(volume).rolling(20).mean() + 1e-8)
        trend_dist = np.abs(close - ema50) / (ema50 + 1e-8)
        f['vol_trend_interaction'] = (v_ratio * trend_dist).fillna(0).values

        # --- New Features: Squeeze and Skew ---
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        f['bb_squeeze'] = np.nan_to_num((upper - lower) / (middle + 1e-8))
        f['vol_skew'] = (pd.Series(high - close) / (pd.Series(close - low) + 1e-8)).clip(0, 5).fillna(1.0).values
        f['rsi_30'] = talib.RSI(close, 30) / 100.0

        # --- Reversion Tension (Continuous) ---
        # Vectorized calculation of Time Since 20d Extremum (Always use integer steps)
        roll_high_20 = pd.Series(high).rolling(20).max()
        roll_low_20 = pd.Series(low).rolling(20).min()
        is_high = (high >= roll_high_20)
        is_low = (low <= roll_low_20)
        idx_series = pd.Series(np.arange(len(close)), index=X.index)
        last_high_idx = pd.Series(np.where(is_high, idx_series, np.nan), index=X.index).ffill()
        last_low_idx = pd.Series(np.where(is_low, idx_series, np.nan), index=X.index).ffill()
        t_since_high = (idx_series - last_high_idx).fillna(0).values
        t_since_low = (idx_series - last_low_idx).fillna(0).values
        t_since_high_log = np.log1p(t_since_high)
        t_since_low_log = np.log1p(t_since_low)
        drop_depth = (roll_high_20 - close) / (close + 1e-8)
        rise_height = (close - roll_low_20) / (roll_low_20 + 1e-8)
        f['roll_tension_hi'] = t_since_high_log * drop_depth.fillna(0).values
        f['roll_tension_lo'] = t_since_low_log * rise_height.fillna(0).values

        # Physics
        log_close = np.log(close + 1e-8)
        peaks, troughs, conf_map = log_move_adaptive_extrema(pd.Series(log_close), **{k:v for k,v in DETECTOR_CONFIG.items() if k != 'method'})
        
        N = len(X)
        last_conf_time, ret_since, tension_hi, tension_lo = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        events_list = sorted([(c, e) for e, c in conf_map.items()])
        curr_e, e_ptr = -1, 0
        for i in range(N):
            while e_ptr < len(events_list) and events_list[e_ptr][0] <= i:
                curr_e = events_list[e_ptr][1]
                e_ptr += 1
            if curr_e != -1:
                dt = i - curr_e
                last_conf_time[i] = np.log1p(dt)
                ret_since[i] = (close[i] - close[curr_e]) / (close[curr_e] + 1e-8)
                window = close[curr_e:i+1]
                p_drop = (np.max(window) - close[i]) / (close[i] * f['atr_14'][i] + 1e-8)
                p_rise = (close[i] - np.min(window)) / (close[i] * f['atr_14'][i] + 1e-8)
                tension_hi[i] = last_conf_time[i] * p_drop
                tension_lo[i] = last_conf_time[i] * p_rise

        f['time_since_last_conf'] = last_conf_time
        f['ret_since_event'] = ret_since
        f['tension_hi'] = tension_hi
        f['tension_lo'] = tension_lo
        
        f['market_alpha'] = np.zeros(len(close))
        f['market_breadth'] = np.zeros(len(close))
        
        if self.request_data:
            alt_rets = []
            basket = ["ethusdt", "bnbusdt", "solusdt", "xrpusdt", "dogeusdt"]
            btc_ret_5 = pd.Series(close).pct_change(5).fillna(0)
            for sym in basket:
                try:
                    extra = self.request_data(sym, X.index)
                    if not extra.empty:
                        if 'timestamp' in extra.columns:
                            extra = extra.set_index('timestamp')
                        e_close = extra.reindex(X.index).close.values
                        alt_rets.append(pd.Series(e_close).pct_change(5).fillna(0).values)
                except: pass
            
            if alt_rets:
                basket_mean = np.mean(alt_rets, axis=0)
                f['market_alpha'] = btc_ret_5.values - basket_mean
                dirs = [np.sign(r) for r in alt_rets]
                f['market_breadth'] = np.mean(dirs, axis=0)

        gate_feats = ['rsi_14', 'mfi_14', 'adx_14', 'vol_intensity', 'vol_regime', 'time_since_last_conf', 'roll_tension_hi', 'roll_tension_lo', 'mom_divergence', 'adx_grad', 'market_alpha', 'market_breadth', 'vol_trend_interaction', 'er_10', 'rsi_vel', 'hurst_100', 'bb_squeeze', 'vol_skew', 'rsi_30']
        for i in range(20 - len(gate_feats)): f[f'g_pad_{i}'] = 0
        hv_feats = ['atr_14', 'log_vol', 'ret_1', 'ret_since_event', 'vol_intensity', 'market_alpha', 'roll_tension_hi', 'roll_tension_lo', 'ema_spread', 'mom_divergence', 'vol_trend_interaction', 'market_breadth', 'bb_squeeze', 'vol_skew', 'rsi_30']
        for i in range(16 - len(hv_feats)): f[f'e_pad_{i}'] = 0
        
        final_df = pd.DataFrame(f, index=X.index).ffill().fillna(0)
        g_cols = gate_feats + [c for c in final_df.columns if c.startswith('g_pad')]
        e_cols = hv_feats + [c for c in final_df.columns if c.startswith('e_pad')]
        other_cols = [c for c in final_df.columns if not (c in g_cols or c in e_cols)]
        return final_df[other_cols + g_cols[:20] + e_cols[:16]]

    def fit(self, df_train, y_train):
        if len(df_train) < self.seq_len: return
        X_f = self.extract_features(df_train)
        valid = ~X_f.isnull().any(axis=1) & ~y_train.isnull().any(axis=1)
        X_clean, y_clean = X_f[valid], y_train[valid]
        self.scaler.fit(X_clean)
        X_s = self.scaler.transform(X_clean)
        y_log = np.log1p(y_clean)
        X_t, y_t = self._create_sequences(X_s, y_log)
        if X_t is None: return
        
        if self.model is None:
            epochs = self.params.get('initial_epochs', 200)
        else:
            epochs = self.params.get('incremental_epochs', 50)
            
        # Create loader
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        
        # Prepare params
        start_date = df_train.timestamp.min().strftime('%Y%m%d%H%M')
        end_date = df_train.timestamp.max().strftime('%Y%m%d%H%M')
        model_path = f"model_weights_{start_date}_{end_date}.pth"
        
        model_params = {
            "input_dim": X_f.shape[1],
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "num_additional_gate_features": 20,
            "num_additional_high_vol_expert_features": 16
        }
        
        train_config = {
            "epochs": epochs,
            "lr": self.learning_rate
        }
        
        self.model, self.optimizer = load_or_train_model(
            model_params=model_params,
            train_loader=loader,
            model_path=model_path,
            device=self.device,
            train_config=train_config,
            model_instance=self.model
        )

    def _create_sequences(self, X, y=None):
        n = len(X) - self.seq_len + 1
        if n <= 0: return None, None
        xs = np.lib.stride_tricks.sliding_window_view(X, (self.seq_len, X.shape[1])).squeeze(1)
        xt = torch.FloatTensor(xs.copy()).to(self.device)
        yt = torch.FloatTensor(y.iloc[self.seq_len-1:].values).to(self.device) if y is not None else None
        return xt, yt

    def predict(self, df, return_regime=False):
        if self.model is None:
            res = (np.zeros((len(df), 2)), np.ones((len(df), 2)))
            return res + (np.zeros((len(df), 3)),) if return_regime else res
        self.model.eval()
        X_f = self.extract_features(df)
        X_s = self.scaler.transform(X_f)
        X_t, _ = self._create_sequences(X_s)
        if X_t is None:
            res = (np.zeros((len(df), 2)), np.ones((len(df), 2)))
            return res + (np.zeros((len(df), 3)),) if return_regime else res
        with torch.no_grad():
            out, gate_weights = self.model(X_t)
            mu = np.expm1(out[:, :2].cpu().numpy().clip(-10, 10))
            sigma = (mu + 1) * torch.sqrt(torch.exp(out[:, 2:])).cpu().numpy() * 1.732
            regime_probs = gate_weights.cpu().numpy()
        pad = np.full((self.seq_len - 1, 2), np.nan)
        mu_full = np.vstack([pad, mu])
        sigma_full = np.vstack([pad, sigma])
        if return_regime:
            pad_regime = np.full((self.seq_len - 1, 3), np.nan)
            return mu_full, sigma_full, np.vstack([pad_regime, regime_probs])
        return mu_full, sigma_full

    def get_trade_decision(self, df_test: pd.DataFrame):
        mu, sigma, regime_probs = self.predict(df_test, return_regime=True)
        from evaluator import log_move_adaptive_extrema, DETECTOR_CONFIG
        log_close = np.log(df_test.close.astype(float))
        peaks, troughs, conf_map = log_move_adaptive_extrema(log_close, **{k:v for k,v in DETECTOR_CONFIG.items() if k != 'method'})
        
        events = sorted([(c, 1 if e in peaks else -1, e) for e, c in conf_map.items()])
        state, time_since_conf, ev_ptr = np.zeros(len(df_test)), np.zeros(len(df_test)), 0
        curr_s, curr_e_idx = 0, -1
        for i in range(len(df_test)):
            while ev_ptr < len(events) and events[ev_ptr][0] <= i:
                curr_s, curr_e_idx = events[ev_ptr][1], events[ev_ptr][2]
                ev_ptr += 1
            state[i] = curr_s
            if curr_e_idx != -1:
                time_since_conf[i] = (df_test.timestamp.iloc[i] - df_test.timestamp.iloc[curr_e_idx]).total_seconds() / 60.0
            
        c_vals, h_vals, l_vals, v_vals = df_test.close.astype(float).values, df_test.high.astype(float).values, df_test.low.astype(float).values, df_test.volume.astype(float).values
        adx, rsi, atr = talib.ADX(h_vals, l_vals, c_vals, 14), talib.RSI(c_vals, 14), talib.ATR(h_vals, l_vals, c_vals, 14)
        adx_sma = talib.SMA(adx, 5)
        atr5, atr20 = talib.ATR(h_vals, l_vals, c_vals, 5), talib.ATR(h_vals, l_vals, c_vals, 20)
        ema20, ema50 = talib.EMA(c_vals, 20), talib.EMA(c_vals, 50)
        rvol = v_vals / (talib.SMA(v_vals, 20) + 1e-8)
        
        # Extended Persistence Logic
        hurst = rolling_hurst(pd.Series(c_vals), window=100).values
        hurst_grad = pd.Series(hurst).diff(5).fillna(0).values
        obv = talib.OBV(c_vals, v_vals)
        obv_corr = pd.Series(obv).rolling(20).corr(pd.Series(c_vals)).fillna(0).values
        
        bp = c_vals - l_vals
        sp = h_vals - c_vals
        pressure_ratio = (pd.Series(bp).rolling(5).mean() / (pd.Series(sp).rolling(5).mean() + 1e-8)).clip(0, 5).fillna(1.0).values
        
        net_chg_ex = pd.Series(c_vals).diff(10).abs()
        abs_chg_sum_ex = pd.Series(c_vals).diff(1).abs().rolling(10).sum()
        er_10 = (net_chg_ex / (abs_chg_sum_ex + 1e-8)).fillna(0).values
        rsi_vel = pd.Series(rsi).diff(3).fillna(0).values
        
        v_ratio = pd.Series(v_vals) / (pd.Series(v_vals).rolling(20).mean() + 1e-8)
        trend_dist = np.abs(c_vals - ema50) / (ema50 + 1e-8)
        vti = (v_ratio * trend_dist).fillna(0).values
        
        mkt_mom, market_breadth = np.zeros(len(df_test)), np.zeros(len(df_test))
        if self.request_data:
            alt_rets = []
            basket = ["ethusdt", "bnbusdt", "solusdt", "xrpusdt", "dogeusdt"]
            btc_ret_5 = pd.Series(c_vals).pct_change(5).fillna(0)
            for sym in basket:
                try:
                    extra = self.request_data(sym, df_test.index)
                    if not extra.empty:
                        if 'timestamp' in extra.columns: extra = extra.set_index('timestamp')
                        e_close = extra.reindex(df_test.index).close.values
                        alt_rets.append(pd.Series(e_close).pct_change(5).fillna(0).values)
                except: pass
            if alt_rets:
                basket_mean = np.mean(alt_rets, axis=0)
                mkt_mom = btc_ret_5.values - basket_mean
                dirs = [np.sign(r) for r in alt_rets]
                market_breadth = np.mean(dirs, axis=0)

        v_ratio_all = v_vals / (talib.SMA(v_vals, 20) + 1e-8)
        trend_dist_signed_all = (c_vals - ema50) / (ema50 + 1e-8)
        tension_all = np.log1p(time_since_conf) * np.abs(trend_dist_signed_all)

        # --- Continuous Tension Calc (Test Time) ---
        roll_high_20_test = pd.Series(h_vals).rolling(20).max()
        roll_low_20_test = pd.Series(l_vals).rolling(20).min()
        is_high_test = (h_vals >= roll_high_20_test)
        is_low_test = (l_vals <= roll_low_20_test)
        idx_series_test = pd.Series(np.arange(len(c_vals)))
        last_high_idx_test = pd.Series(np.where(is_high_test, idx_series_test, np.nan)).ffill()
        last_low_idx_test = pd.Series(np.where(is_low_test, idx_series_test, np.nan)).ffill()
        t_since_high_test = (idx_series_test - last_high_idx_test).fillna(0).values
        t_since_low_test = (idx_series_test - last_low_idx_test).fillna(0).values
        roll_tension_hi = np.log1p(t_since_high_test) * ((roll_high_20_test - c_vals) / (c_vals + 1e-8)).fillna(0).values
        roll_tension_lo = np.log1p(t_since_low_test) * ((c_vals - roll_low_20_test) / (roll_low_20_test + 1e-8)).fillna(0).values

        signals, sl = np.zeros(len(df_test)), np.zeros(len(df_test))
        f_counts = {"Sig":0, "Rem":0, "ADX":0, "Vol":0, "ER":0, "Tens":0, "Div":0, "Breadth":0, "VTI":0, "Fractal":0}
        boost_count = 0
        fast_exit_count = 0
        
        # RAP Diagnostics
        rap_sigma_rescued = 0
        rap_accum_rescued = 0
        
        for i in range(self.seq_len, len(df_test)):
            t_total, s_next = mu[i, 0], sigma[i, 0]
            # --- Regime Detection ---
            # gate_weights: [Trending, Ranging, HighVol]
            p_regime = regime_probs[i]
            if np.isnan(p_regime[0]): continue
            regime_idx = np.argmax(p_regime)
            
            alpha_dir = 1 if state[i] == -1 else -1 
            
            vol_shock = atr5[i] / (atr20[i] + 1e-8)
            tension = tension_all[i]
            adx_factor = max(1.0, adx[i] / 30.0)

            # Micro-structure Ignition Detect (Alpha Confirmed)
            is_igniting_long = pressure_ratio[i] > self.ignition_thr and obv_corr[i] > 0.35 and mkt_mom[i] > 0 and vol_shock < 2.0
            is_igniting_short = pressure_ratio[i] < (1.0 / self.ignition_thr) and obv_corr[i] > 0.35 and mkt_mom[i] < 0 and vol_shock < 2.0
            is_igniting = (alpha_dir == 1 and is_igniting_long) or (alpha_dir == -1 and is_igniting_short)
            
            # 1. Budget Modulation & Regime Adaptation
            budget_mult = 1.0
            if hurst[i] > 0.60 or hurst_grad[i] > 0.02: 
                budget_mult += 0.25
            elif is_igniting:
                budget_mult += 0.20 
            elif hurst[i] < 0.50: 
                budget_mult -= 0.15
            
            # Regime-Specific Modulation
            dyn_rsi_upper, dyn_rsi_lower, dyn_sigma_mult = self.rsi_upper, self.rsi_lower, 1.0
            if regime_idx == 0: # Trending Specialist
                budget_mult += 0.20
                dyn_rsi_upper, dyn_rsi_lower = 82, 18
                dyn_sigma_mult = 1.10
            elif regime_idx == 1: # Ranging Specialist
                budget_mult -= 0.10
                dyn_rsi_upper, dyn_rsi_lower = 72, 28
                dyn_sigma_mult = 0.95
            else: # High Vol Specialist (Defensive)
                budget_mult -= 0.40
                dyn_rsi_upper, dyn_rsi_lower = 65, 35
                dyn_sigma_mult = 0.85
            
            # Simplified Elasticity Boost
            current_tension = roll_tension_lo[i] if alpha_dir == 1 else roll_tension_hi[i]
            if current_tension > self.tension_gate:
                 budget_mult += self.tension_boost
                 boost_count += 1
            
            if obv_corr[i] > self.obv_corr_gate: budget_mult += 0.10
            
            # RAP: ADX-Dynamic Sigma Scaling
            base_budget = budget_mult
            if self.adx_sigma_slope > 0:
                # Pivot at 20.0 instead of 25.0 to be less restrictive
                adx_contrib = (adx[i] - 20.0) * self.adx_sigma_slope
                # Only restrict budget if trend is NOT aligned
                if adx_contrib < 0:
                    budget_mult += adx_contrib
                else:
                    # Positive contribution requires trend alignment
                    if (alpha_dir == 1 and c_vals[i] > ema20[i]) or (alpha_dir == -1 and c_vals[i] < ema20[i]):
                        budget_mult += adx_contrib

            eff_sigma_thresh = self.sigma_thresh * budget_mult * dyn_sigma_mult
            
            if not np.isnan(t_total) and s_next > eff_sigma_thresh:
                f_counts["Sig"] += 1
                continue
            
            # Track if RAP-Sigma actually allowed a trade that the base budget would have rejected
            if not np.isnan(t_total) and s_next > (self.sigma_thresh * base_budget):
                rap_sigma_rescued += 1
            
            if np.isnan(t_total):
                continue
            
            # 2. Dynamic Runway (Compression on Acceleration)
            runway_mult = 1.0
            if hurst_grad[i] > 0.01: runway_mult = 0.80
            if is_igniting: runway_mult = 0.75 
            
            eff_t_min = self.t_remaining_min * runway_mult
            t_remaining = t_total - time_since_conf[i]
            if t_remaining < eff_t_min: f_counts["Rem"] += 1; continue
            
            # 3. Persistence Floor vs. Acceleration Override (Selective Bypass)
            eff_hurst_floor = self.hurst_thresh - (0.05 if is_igniting else 0.0)
            
            # RAP: Accumulation Bypass
            is_accumulation = False
            if self.accum_bypass:
                # Accumulation: Quiet Volatility + Strong Pressure
                is_accum_long = (alpha_dir == 1 and pressure_ratio[i] > 1.25 and vol_shock < 1.0)
                is_accum_short = (alpha_dir == -1 and pressure_ratio[i] < 0.8 and vol_shock < 1.0)
                is_accumulation = (is_accum_long or is_accum_short)

            if hurst[i] < eff_hurst_floor and hurst_grad[i] < 0.01: 
                if not is_accumulation:
                    f_counts["Fractal"] += 1; continue
                else:
                    rap_accum_rescued += 1
            
            # ADX Lag Bypass for ignition
            if not is_igniting:
                if i > 5 and adx[i] < adx_sma[i] and adx[i] > 35: f_counts["ADX"] += 1; continue
            
            # --- Tension & ER Logic (HARDENED) ---
            dyn_tension_limit = self.tension_limit * adx_factor
            dyn_er_thresh = self.er_thresh / (1.0 + 0.3 * min(vol_shock, 3.0))

            is_alpha = (mkt_mom[i] * alpha_dir) > 0
            dyn_div_limit = self.divergence_thresh * (3.0 if is_alpha else 1.0)

            if vol_shock > 2.5: f_counts["Vol"] += 1; continue 
            if er_10[i] < dyn_er_thresh: f_counts["ER"] += 1; continue 
            if tension > dyn_tension_limit: f_counts["Tens"] += 1; continue 
            if np.abs(mkt_mom[i]) > dyn_div_limit: f_counts["Div"] += 1; continue 
            
            # Robust VTI Filter 
            vti_signed = v_ratio_all[i] * trend_dist_signed_all[i]
            
            if alpha_dir == 1 and vti_signed > self.vti_limit and obv_corr[i] < self.obv_corr_gate:
                f_counts["VTI"] += 1; continue
            if alpha_dir == -1 and vti_signed < -self.vti_limit and obv_corr[i] < self.obv_corr_gate:
                f_counts["VTI"] += 1; continue
            
            if self.use_breadth:
                 if alpha_dir == 1 and market_breadth[i] < -self.breadth_gate: f_counts["Breadth"] += 1; continue
                 if alpha_dir == -1 and market_breadth[i] > self.breadth_gate: f_counts["Breadth"] += 1; continue

            close = c_vals[i]
            is_uptrend = (close > ema50[i]) and (close > ema20[i])
            is_downtrend = (close < ema50[i]) and (close < ema20[i])
            is_strong_trend = adx[i] > self.adx_thresh
            
            # Entry Microstructure Trigger
            allow_trade = (vol_shock < 1.8) or (is_strong_trend and rvol[i] > 1.2) or is_igniting
            
            # --- Stateful Position Management ---
            if i > 0: current_pos = signals[i-1]
            else: current_pos = 0
            
            new_signal = 0
            
            # 1. Check Entry (with Dynamic RSI)
            entry_long = False
            entry_short = False
            
            if allow_trade:
                if state[i] == -1: # Seeking Peak (Bullish Setup)
                    if (is_uptrend and is_strong_trend and pressure_ratio[i] > self.pressure_trigger and rsi[i] > 45) or is_igniting_long:
                        if rsi[i] < dyn_rsi_upper and rsi_vel[i] > -2: 
                            entry_long = True
                elif state[i] == 1: # Seeking Trough (Bearish Setup)
                    if (is_downtrend and is_strong_trend and pressure_ratio[i] < (1.0 / self.pressure_trigger) and rsi[i] < 55) or is_igniting_short:
                        if rsi[i] > dyn_rsi_lower and rsi_vel[i] < 2: 
                            entry_short = True
            
            # 2. Manage State (Regime-Adaptive Exits)
            if current_pos == 0:
                if entry_long: new_signal = 1
                elif entry_short: new_signal = -1
            elif current_pos == 1: # Long
                if entry_short: new_signal = -1 # Flip
                else:
                    if regime_idx == 0: # Trending: Patience
                        exit_cond = (close < ema50[i] and rsi[i] < 42) or (rsi[i] > 92)
                    elif regime_idx == 1: # Ranging: Precision
                        exit_cond = (close < ema20[i] and rsi[i] < 48) or (rsi[i] > 78)
                    else: # High Vol: Safety
                        exit_cond = (close < ema20[i]) or (rsi[i] > 75)
                    
                    # NEW: Fast Exit for Stretched positions
                    if roll_tension_lo[i] > self.exit_tension_gate and close < ema20[i]:
                        exit_cond = True
                        fast_exit_count += 1
                    
                    if exit_cond: new_signal = 0
                    else: new_signal = 1 # Hold
            elif current_pos == -1: # Short
                if entry_long: new_signal = 1 # Flip
                else:
                    if regime_idx == 0: # Trending
                        exit_cond = (close > ema50[i] and rsi[i] > 58) or (rsi[i] < 8)
                    elif regime_idx == 1: # Ranging
                        exit_cond = (close > ema20[i] and rsi[i] > 52) or (rsi[i] < 22)
                    else: # High Vol
                        exit_cond = (close > ema20[i]) or (rsi[i] < 25)
                    
                    # NEW: Fast Exit for Stretched positions
                    if roll_tension_hi[i] > self.exit_tension_gate and close > ema20[i]:
                        exit_cond = True
                        fast_exit_count += 1

                    if exit_cond: new_signal = 0
                    else: new_signal = -1 # Hold
            
            signals[i] = new_signal
            
            if not np.isnan(atr[i]):
                sl_mult = 1.2 if hurst[i] > 0.6 else 0.8
                
                # NEW: Adaptive Tension SL
                if (new_signal == 1 and roll_tension_lo[i] > self.exit_tension_gate) or \
                   (new_signal == -1 and roll_tension_hi[i] > self.exit_tension_gate):
                    sl_mult *= 0.75
                
                sl[i] = (self.atr_mult_base * sl_mult * atr[i] / close) * (1 + s_next/1500)
            else: sl[i] = 0.02
            
        print(f"Decisions: {int(np.sum(signals!=0))} Signal Events (Boosted: {boost_count}, FastExit: {fast_exit_count}, RAP-Sigma: {rap_sigma_rescued}, RAP-Accum: {rap_accum_rescued}). Filtered: {f_counts}")
        return pd.DataFrame({'signal': signals, 'stop_loss_pct': sl}, index=df_test.index)