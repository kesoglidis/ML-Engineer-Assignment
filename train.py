"""
Train multiple RL agents (A2C, DQN) on the GridEnv environment.
Training continues until convergence, then models are saved to disk.
"""

import os
import numpy as np
from expanded_env import GridEnv, IntroduceBonusesCallback
from stable_baselines3 import A2C, DQN
from evaluate import evaluate


def has_converged(reward_history, patience=5, min_delta=0.01):
    """Return True if recent average reward shows little or no improvement."""
    if len(reward_history) < patience * 2:
        return False
    recent = np.array(reward_history[-patience:])
    previous = np.array(reward_history[-(2 * patience):-patience])
    return abs(recent.mean() - previous.mean()) < min_delta and previous.mean() > 0 



def train(env, model_class, algo_name, max_steps=1_000_000, eval_interval=1000, save_dir="trained_models"):
    """Train a Stable Baselines3 model until convergence, then save it."""
    print(f"Training {algo_name} until convergence")

    model = model_class("MlpPolicy", env, verbose=0, device='cuda')
    # if "DQN" in algo_name:
    #     model = model_class("MlpPolicy", env, verbose=0, learning_rate=1e-4, device='cuda')
    # elif "A2C" in algo_name:
        
    # elif "A2C" in algo_name:
    #     model = model_class("MultiInputPolicy", env, verbose=0, device = 'cuda')
    #                         # learning_rate=1e-3, n_steps=32, gae_lambda=0.95, 
    #                         # ent_coef=0.2, vf_coef=0.5, max_grad_norm = 0.5, 
    #                         # device='cuda')
    # elif 'DQN' in algo_name:
    #     model = model_class("MultiInputPolicy", env, verbose=0, device = 'cuda')
        # learning_rate=1e-4,
        #                     buffer_size=100_000, train_freq = 8,
        #                     target_update_interval=2000, exploration_fraction=0.3, device='cuda')

    reward_history = []
    total_steps = 0
    success_rate = 0
    os.makedirs(save_dir, exist_ok=True)

    while total_steps < max_steps:

        # callback = IntroduceBonusesCallback(success_rate, success_threshold=70, verbose=1)
        # model.learn(total_timesteps=eval_interval, reset_num_timesteps=False, callback=callback)

        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        total_steps += eval_interval

        avg_reward, outcomes, avg_steps, success_rate = evaluate(model, env, n_episodes=100)

        reward_history.append(avg_reward)
        print(f"[{algo_name}] Steps: {total_steps:6d} | Avg Reward: {avg_reward:8.3f} \
              | Avg Steps: {avg_steps:8.1f} | Success Rate: {success_rate}")
        for k, v in outcomes.items():
            print(f"{k.capitalize()}: {v}")
        if has_converged(reward_history):# and callback.done():
            print(f"{algo_name} converged after {total_steps} timesteps.")
            break
        
    # Final evaluation
    avg_reward, outcomes, avg_steps , success_rate  = evaluate(model, env, n_episodes=1000)
    print(f"{algo_name} Final Results after {total_steps} timesteps")
    print(f"Avg Reward: {avg_reward:8.3f} \
              | Avg Steps: {avg_steps:8.1f} | Success Rate: {success_rate}")
    for k, v in outcomes.items():
        print(f"{k.capitalize()}: {v}")

    # Save trained model
    save_path = os.path.join(save_dir, f"{algo_name}_final.zip")
    model.save(save_path)
    print(f"{algo_name} model saved to {save_path}\n")

    return model, reward_history


if __name__ == "__main__":
    # Define the algorithms to train
    algorithms = {
        "A2C": A2C,
        "DQN": DQN,
    }

    # Simple Grid Environment (obstacles only) 
    # simple_envs = {
    #     "A2C": GridEnv(width=6, height=6, obstacles=5, bonus=0),
    #     "DQN": GridEnv(width=6, height=6, obstacles=5, bonus=0),
    # }

    # print("Training on Grid Environment with obstacles only (single target)")
    # for name, algo_class in algorithms.items():
    #     train(simple_envs[name], algo_class, name)

    # Expanded Grid Environment (bonuses included)
    expanded_envs = {
        "A2C": GridEnv(width=6, height=6, obstacles=5, bonus=2),
        "DQN": GridEnv(width=6, height=6, obstacles=5, bonus=2),
    }

    print("Training on Expanded Grid Environment with bonus positions")
    for name, algo_class in algorithms.items():
        train(expanded_envs[name], algo_class, f"Expanded_b2_{name}", eval_interval=5000)

    expanded_envs = {
        "A2C": GridEnv(width=6, height=6, obstacles=5, bonus=5),
        "DQN": GridEnv(width=6, height=6, obstacles=5, bonus=5),
    }

    print("Training on Expanded Grid Environment with bonus positions")
    for name, algo_class in algorithms.items():
        train(expanded_envs[name], algo_class, f"Expanded_b5_{name}", eval_interval=5000)
