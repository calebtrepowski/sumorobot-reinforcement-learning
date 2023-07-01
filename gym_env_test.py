import numpy as np
from gym_env import SumoRobotEnv

env = SumoRobotEnv(render_mode="human")
observation = env.reset()

for i in range(1000):
    action = env.action_space.sample()

    observation, reward, terminated, info = env.step(action)

    if terminated:
        observation = env.reset()


env.close()
