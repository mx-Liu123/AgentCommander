import numpy as np
import pandas as pd

def calculate_rl_score(history: list) -> float:
    """
    Contracted Brain: Calculate a scalar score for RL interaction history.
    Input: history (list of dicts from env.step info).
    Output: Return a float. Standard is 'Higher is Better' (e.g. Total Reward, Sharpe).
    """
    if not history: return 0.0
    df = pd.DataFrame(history)
    
    # AI: Implement specific scoring logic (e.g., final equity, mean reward)
    score = 0.0
    if 'reward_step' in df.columns:
        score = df['reward_step'].sum()
        
    return float(score)
