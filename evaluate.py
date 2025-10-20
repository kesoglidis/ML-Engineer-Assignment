import os
import numpy as np
from expanded_env import GridEnv
from dqn_environment import DQNGridEnv
from collections import defaultdict
from stable_baselines3 import A2C, DQN, PPO


def evaluate(model, env, n_episodes=100, render=False):
    """Evaluate a trained model over multiple episodes."""
    vec_env = model.get_env()
    obs = vec_env.reset()

    total_rewards = []
    total_steps = []
    outcomes = defaultdict(int)
    outcomes["bonus"] = 0

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
                    done = False
                    # outcomes["unknown"] += 1
                if "hit_bonus" in info:
                    outcomes["bonus"] += info["hit_bonus"]
                episode_reward = reward

        total_rewards.append(episode_reward)
        total_steps.append(steps)

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = (outcomes["targets"] / n_episodes) * 100.0

    return avg_reward, outcomes, avg_steps, success_rate # avg bonus #time to first bonus


if __name__ == "__main__":
    model_dir = "trained_models"

    # Define environments per algorithm
    envs = {
        "A2C": GridEnv(width=6, height=6, obstacles=5, bonus=0),
        "DQN": GridEnv(width=6, height=6, obstacles=5, bonus=0),
        "Expanded_b2_A2C": GridEnv(width=6, height=6, obstacles=5, bonus=2),
        "Expanded_b2_DQN": GridEnv(width=6, height=6, obstacles=5, bonus=2),
        "Expanded_b5_A2C": GridEnv(width=6, height=6, obstacles=5, bonus=2),
        "Expanded_b5_DQN": GridEnv(width=6, height=6, obstacles=5, bonus=2),
    }

    # Define algorithms to evaluate
    algorithms = {
        "A2C": A2C,
        "DQN": DQN,
        "Expanded_b2_A2C": A2C,
        "Expanded_b2_DQN": DQN,
        "Expanded_b5_A2C": A2C,
        "Expanded_b5_DQN": DQN,
    }

    print("Agent Evaluation Results (100 Episodes)\n")

    for name, algo_class in algorithms.items():
        env = envs[name]  # map name to env
        model_path = os.path.join(model_dir, f"{name}_final.zip")
        
        # Load the trained model
        model = algo_class.load(model_path, env=env)

        # Evaluate
        avg_reward, outcomes, avg_steps, success_rate = evaluate(model, env, n_episodes=100)

        # Print results
        print(f"\n{name}")
        print(f"Average steps per episode:    {avg_steps:.2f}")
        print(f"Average total reward:         {avg_reward:.3f}")
        print(f"Success rate (hit target %): {success_rate:.2f}%")
        for k, v in outcomes.items():
            print(f"{k.capitalize()}: {v}")

            
