import numpy as np
from gym_env import SumoRobotEnv

env = SumoRobotEnv(render_mode="human")
observation, info = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    action = np.abs(action)

    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
