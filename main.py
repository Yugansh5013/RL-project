import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from custom_acrobot import CustomAcrobotEnv


env = gym.make("Acrobot-v1", render_mode="rgb_array")


#state=env.reset()
# image=env.render()
# plt.imshow(image)
# plt.show()
# plt.plot(state)
env = RecordVideo(env, video_folder='./video', episode_trigger=lambda x: x % 1 == 0)

# Reset the environment and get initial state
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Take a random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Ensure rendering happens
    env.render()
    
    if terminated or truncated:
        break

env.close()

