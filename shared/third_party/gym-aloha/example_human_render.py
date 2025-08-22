import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
from tqdm import tqdm

env = gym.make("gym_aloha/AlohaDummyInsertion-v0", render_mode="human")
observation, info = env.reset()

for _ in tqdm(range(1000)):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()
