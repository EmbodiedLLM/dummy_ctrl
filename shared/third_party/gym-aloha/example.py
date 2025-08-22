import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
from tqdm import tqdm

env = gym.make("gym_aloha/AlohaDummyInsertion-v0")
observation, info = env.reset()
frames = []

for _ in tqdm(range(100)):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
