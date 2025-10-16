"""
Train multiple RL agents (A2C, DQN) on the GridEnv environment.
Training continues until convergence, then models are saved to disk.
"""

import os
import numpy as np
from collections import defaultdict
from environment import GridEnv
from stable_baselines3 import A2C, DQN, PPO
from evaluate import evaluate


def has_converged(reward_history, patience=5, min_delta=0.01):
    """Return True if recent average reward shows little or no improvement."""
    if len(reward_history) < patience * 2:
        return False
    recent = np.array(reward_history[-patience:])
    previous = np.array(reward_history[-(2 * patience):-patience])
    return abs(recent.mean() - previous.mean()) < min_delta



def train(env, model_class, algo_name, max_steps=200_000_0, eval_interval=1000, save_dir="trained_models"):
    """Train a Stable Baselines3 model until convergence, then save it."""
    print(f"Training {algo_name} until convergence")


    model = model_class("MultiInputPolicy", env, verbose=0, learning_rate=1e-4)
    reward_history = []
    total_steps = 0

    os.makedirs(save_dir, exist_ok=True)

    while total_steps < max_steps:
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        total_steps += eval_interval

        avg_reward, outcomes, avg_steps, success_rate = evaluate(model, env, n_episodes=100)
        reward_history.append(avg_reward)
        print(f"[{algo_name}] Steps: {total_steps:6d} | Avg Reward: {avg_reward:8.3f} \
              | Avg Steps: {avg_steps:8.1f} | Sucess Rate: {success_rate}")

        if has_converged(reward_history):
            print(f"{algo_name} converged after {total_steps} timesteps.")
            break

    # Final evaluation
    avg_reward, outcomes, avg_steps , success_rate  = evaluate(model, env, n_episodes=1000)
    print(f"{algo_name} Final Results after {total_steps} timesteps")
    print(f"Avg Reward: {avg_reward:8.3f} \
              | Avg Steps: {avg_steps:8.1f} | Sucess Rate: {success_rate}")
    for k, v in outcomes.items():
        print(f"{k.capitalize()}: {v}")

    # Save trained model
    save_path = os.path.join(save_dir, f"{algo_name}_final.zip")
    model.save(save_path)
    print(f"{algo_name} model saved to {save_path}\n")

    return model, reward_history


if __name__ == "__main__":
    env = GridEnv(width=6, height=6, obstacles=5)

    algorithms = {
        "A2C": A2C,
        "DQN": DQN,
        #"PPO": PPO,
    }

    for name, algo_class in algorithms.items():
        train(env, algo_class, name)
