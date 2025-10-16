"""
Evaluate trained RL agents (A2C, DQN, PPO) on the GridEnv environment.
Computes average episode length, average reward, and success rate.
"""

import os
import numpy as np
from environment import GridEnv
from collections import defaultdict
from stable_baselines3 import A2C, DQN, PPO


def evaluate(model, env, n_episodes=100, render=False):
    """Evaluate a trained model over multiple episodes."""
    vec_env = model.get_env()
    obs = vec_env.reset()

    total_rewards = []
    total_steps = []
    outcomes = defaultdict(int)

    for ep in range(n_episodes):
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            info = info[0]
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1

            if done:
                if info.get("hit_target"):
                    outcomes["targets"] += 1
                elif info.get("hit_obstacle"):
                    outcomes["obstacles"] += 1
                elif info.get("hit_wall"):
                    outcomes["walls"] += 1
                elif info.get("TimeLimit.truncated"):
                    outcomes["timelimits"] += 1
                else:
                    outcomes["unknown"] += 1
                # episode_reward = reward
            # if render:
            #     vec_env.render("human")

        total_rewards.append(episode_reward)
        total_steps.append(steps)

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = (outcomes["targets"] / n_episodes) * 100.0

    return avg_reward, outcomes, avg_steps, success_rate


if __name__ == "__main__":
    env = GridEnv()
    model_dir = "trained_models"
    algorithms = {
        "A2C": A2C,
        "DQN": DQN,
        #"PPO": PPO,
    }

    print("Agent Evaluation Results (100 Episodes)")

    for name, algo_class in algorithms.items():
        model_path = os.path.join(model_dir, f"{name}_final.zip")
        if not os.path.exists(model_path):
            print(f"Model for {name} not found at {model_path}, skipping.")
            continue

        model = algo_class.load(model_path, env=env)
        avg_steps, avg_reward, success_rate = evaluate(model, env, n_episodes=100)

        print(f"\n--- {name} ---")
        print(f"Average steps per episode:   {avg_steps:.2f}")
        print(f"Average total reward:        {avg_reward:.3f}")
        print(f"Success rate (hit target %): {success_rate:.2f}%")
