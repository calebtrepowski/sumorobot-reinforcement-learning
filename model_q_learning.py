import numpy as np
from typing import Iterable
from gym_env import SumoRobotEnv
import matplotlib.pyplot as plt

env = SumoRobotEnv(render_mode="human")

# Initialize Q-table
num_observations = 8
num_actions = 4
q_table = np.zeros((2**num_observations, num_actions))

# Hyperparameters
num_episodes = 10000
max_iter_episode = 250
initial_exploration_proba = 0.6
exploration_proba = initial_exploration_proba
exploration_decreasing_decay = 0.01
min_exploration_proba = 0.01
gamma = 0.99
lr = 0.1

rewards_per_episode = []


def binary_state_to_integer(state: Iterable):
    return int("".join(map(str, state)), 2)


try:
    for episode in range(num_episodes):
        current_state = env.reset()
        done = False

        total_episode_reward = 0

        for i in range(max_iter_episode):
            current_state_index = binary_state_to_integer(current_state)

            if np.random.uniform(0, 1) < exploration_proba:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[current_state_index])

            next_state, reward, done, _ = env.step(action)
            next_state_index = binary_state_to_integer(next_state)

            q_table[current_state_index, action] = (
                1-lr) * q_table[current_state_index, action] + lr*(reward + gamma*np.max(q_table[next_state_index]))

            total_episode_reward = total_episode_reward + reward

            if done:
                break
            current_state = next_state

        exploration_proba = max(min_exploration_proba,
                                initial_exploration_proba*np.exp(-exploration_decreasing_decay*episode))
        rewards_per_episode.append(total_episode_reward)

        if episode % 100 == 0:
            print(f"Episode: {episode}")
except:
    pass
finally:
    np.save("weights/q_table.npy", q_table)
    env.close()


plt.figure()
plt.plot(range(len(rewards_per_episode)), rewards_per_episode)
plt.title("Rewards por episodio")
plt.xlabel("Episodio")
plt.ylabel("Reward")
plt.show()

# # Evaluate the learned policy
# num_eval_episodes = 10
# total_rewards = 0

# for _ in range(num_eval_episodes):
#     current_state = env.reset()
#     done = False

#     while not done:
#         current_state_index = binary_state_to_integer(current_state)
#         action = np.argmax(q_table[current_state_index])
#         current_state, reward, done, _ = env.step(action)
#         total_rewards += reward

# average_reward = total_rewards / num_eval_episodes
# print(
#     f"Average reward over {num_eval_episodes} evaluation episodes: {average_reward}")
