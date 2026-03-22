import os
import sys
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# 路径对齐：确保能够加载 strategy_lib/ 文件夹下的依赖
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "strategy_lib"))

from strategy_lib.env import NautilusEnv
from strategy_lib.visualizer import TradingVisualizer

class EvalAndSaveCallbackEOD(BaseCallback):
    def __init__(self, eval_env, total_steps, steps_per_ep):
        super(EvalAndSaveCallbackEOD, self).__init__(0)
        self.eval_env = eval_env
        self.total_steps = total_steps
        self.steps_per_ep = steps_per_ep
        self.episode_count = 0
        self.best_det_profit = -999.0 
        self.current_f_scale = -1.0 # 追踪当前等级
        self.found_positive_in_scale = False # 标记当前等级是否已攻克
        self.start_time = time.time()
        self.model_save_path = "models/best_model_eod"
        self.log_file = "train_debug_eod.log"

    def _on_step(self) -> bool:
        total_episodes = max(self.total_steps / self.steps_per_ep, 4)
        current_ep = self.num_timesteps / self.steps_per_ep
        
        # 动态比例切换 (保证每个阶段至少有 1 个 ep)
        if current_ep < (total_episodes * 0.3): f_scale = 0.0
        elif current_ep < (total_episodes * 0.6): f_scale = 0.2
        elif current_ep < (total_episodes * 0.8): f_scale = 0.5
        else: f_scale = 1.0
        
        # 检测等级切换
        if f_scale > self.current_f_scale:
            self.current_f_scale = f_scale
            self.found_positive_in_scale = False
            print(f"\n>>> [LEVEL UP] Friction Scale increased to {f_scale}. Resetting local baseline...")

        self.training_env.set_attr("friction_scale", f_scale)
        self.model.ent_coef = max(0.1 - (self.num_timesteps/self.total_steps) * 0.09, 0.01)

        if "dones" in self.locals and self.locals["dones"][0]:
            self.episode_count += 1
            self.eval_env.friction_scale = self.current_f_scale # 同步评估等级
            det_profit, det_trades = self._run_deterministic_eval()
            
            should_save = False
            status = " [KEEP]"
            
            if det_profit > 0:
                if not self.found_positive_in_scale:
                    # 规则 A：当前等级首次出现正收益，无视旧数值，强制设为新基准
                    should_save = True
                    self.found_positive_in_scale = True
                    status = " [NEW SCALE WINNER]"
                elif det_profit > self.best_det_profit:
                    # 规则 B：在当前已攻克的等级中，寻找更优解
                    should_save = True
                    status = " [AUDIT BEST SAVED]"
            else:
                # 规则 C：如果当前等级全是负收益，则只在比历史最高（可能来自旧等级）还好时才更新
                if not self.found_positive_in_scale and det_profit > self.best_det_profit:
                    should_save = True
                    status = " [BEST NEGATIVE]"

            if should_save:
                self.best_det_profit = det_profit
                self.model.save(self.model_save_path)
            
            msg = f">>> EOD_EP {self.episode_count:02d} | Det Audit: {det_profit:+.2f}% ({det_trades} trades) | Fric: {f_scale:.1f}{status}"
            print(msg)
            with open(self.log_file, "a") as f: f.write(msg + "\n")
        return True

    def _run_deterministic_eval(self):
        obs, _ = self.eval_env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.eval_env.step(action)
        return (info.get('equity', 1e6)-1e6)/1e4, info.get('trades', 0)

def run_eod_evolution():
    print(f"=== [Nautilus PORTABLE EOD RUNNER] ===\n")
    # 灵活配置 Episode 数量，保底 4 个
    n_eps = int(os.environ.get("TOTAL_EPISODES", 4))
    n_eps = max(n_eps, 4)
    
    env = NautilusEnv()
    eval_env = NautilusEnv()
    total_steps = env.total_data_steps * n_eps
    print(f"Starting Curriculum Learning with {n_eps} episodes ({total_steps} total steps)...")
    
    model = PPO("MlpPolicy", env, verbose=0, n_steps=4096, device="cpu")
    callback = EvalAndSaveCallbackEOD(eval_env, total_steps, env.total_data_steps)
    model.learn(total_timesteps=total_steps, callback=callback)
    
    print("Generating EOD Audit Plots...")
    best_model = PPO.load("models/best_model_eod.zip")
    obs, _ = eval_env.reset()
    done, history = False, []
    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        mid_price = (eval_env._ask_series[eval_env.current_step-1] + eval_env._bid_series[eval_env.current_step-1])/2.0
        history.append({'timestamp': info['timestamp'], 'price': float(mid_price), 'equity': info['equity'], 'position': info['position'], 'action': int(action)})
    TradingVisualizer.plot_daily_results(pd.DataFrame(history), base_dir="plots_eod")
    
    # 关键：为 AgentCommander 打印标准化指标
    print(f"\nFinal Audit Complete.")
    print(f"Best metric: {callback.best_det_profit:.6f}")

if __name__ == "__main__":
    run_eod_evolution()
