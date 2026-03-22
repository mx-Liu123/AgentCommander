import matplotlib.pyplot as plt
import pandas as pd
import os

def draw_rl_plots(history: list, output_dir: str):
    """
    Contracted Eyes: Generate visualizations for RL interaction history.
    """
    if not history: return
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(history)
    
    # AI: Implement visualization logic (e.g., Cumulative Reward Plot)
    plt.figure(figsize=(10, 6))
    if 'reward_step' in df.columns:
        plt.plot(df['reward_step'].cumsum(), label='Cumulative Reward')
        plt.title("RL Agent Performance")
        plt.xlabel("Step")
        plt.ylabel("Reward Sum")
        plt.legend()
        
    save_path = os.path.join(output_dir, "best_result.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")
