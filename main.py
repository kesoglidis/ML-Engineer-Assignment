from typing import Optional
import numpy as np
import gymnasium as gym
import tqdm

from environment import GridEnv
from stable_baselines3 import A2C

env = GridEnv()

model = A2C("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
