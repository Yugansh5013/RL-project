import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

# Create and wrap the environment
env = gym.make("Acrobot-v1", render_mode="rgb_array")
env = RecordVideo(env, "videos", episode_trigger=lambda x: x % 1 == 0)  # Record every episode

# Number of episodes to run
n_episodes = 3

for episode in range(n_episodes):
    observation, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        # For this example, we'll just use random actions
        action = env.action_space.sample()
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    
    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

env.close()
print("Simulation completed. Check the 'videos' directory for the recorded episodes.") 