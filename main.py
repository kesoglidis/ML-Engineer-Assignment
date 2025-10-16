from typing import Optional
import numpy as np
import gymnasium as gym
from collections import defaultdict

from environment import GridEnv
from stable_baselines3 import A2C, DQN, PPO


def evaluate_model(model, env, n_episodes=1000, render=False):
    """Evaluate a trained model and return outcome statistics."""
    vec_env = model.get_env()
    obs = vec_env.reset()
    outcomes = defaultdict(int)

    for _ in range(n_episodes):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        info = info[0]

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

        if render:
            vec_env.render("human")

    return outcomes


def print_outcomes_summary(algorithm_name, steps, outcomes):
    print(f"\n===== {algorithm_name} Results after {steps} timesteps =====")
    for k, v in outcomes.items():
        print(f"{k.capitalize()}: {v}")


def train_and_evaluate(env, model_class, algo_name, time_steps_list):
    """Train and evaluate a given RL model for multiple time step values."""
    for steps in time_steps_list:
        print(f"\n=== Training {algo_name} for {steps} timesteps ===")
        model = model_class("MultiInputPolicy", env, verbose=0)
        model.learn(total_timesteps=steps)
        outcomes = evaluate_model(model, env, n_episodes=1000)
        print_outcomes_summary(algo_name, steps, outcomes)


if __name__ == "__main__":
    env = GridEnv(width=6, height=6, obstacles=5)

    # Algorithms to test
    algorithms = {
        "A2C": A2C,
        "DQN": DQN,
        #"PPO": PPO,
    }

    # Different training lengths
    time_steps_list = [2000, 20000, 200000]#50000, 100000]

    for name, algo_class in algorithms.items():
        train_and_evaluate(env, algo_class, name, time_steps_list)
