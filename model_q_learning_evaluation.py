from gym_env import SumoRobotEnv
from typing import Iterable
import numpy as np

env = SumoRobotEnv(render_mode="human")

q_table = np.load("weights/q_table.npy")

# Evaluate the learned policy
num_eval_episodes = 50
total_rewards = 0

def binary_state_to_integer(state: Iterable):
    return int("".join(map(str, state)), 2)

for _ in range(num_eval_episodes):
    current_state = env.reset()
    done = False

    while not done:
        current_state_index = binary_state_to_integer(current_state)
        action = np.argmax(q_table[current_state_index])
        current_state, reward, done, _ = env.step(action)
        total_rewards += reward

average_reward = total_rewards / num_eval_episodes
print(
    f"Average reward over {num_eval_episodes} evaluation episodes: {average_reward}")
