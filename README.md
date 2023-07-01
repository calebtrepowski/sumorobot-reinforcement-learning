# Sumorobot Reinforcement Learning

Implementation of SumoRobot environment, based on gymnasium environments. Used for training Q-Learning agent.

## Environment Description

Observation Space: `MultiBinary(8)`. 6 proximity sensors, 2 line sensors.

Action Space: `Discrete(4)`. 4 possible actions: go forward, go back, turn left, turn right.

## Demos

Environment demo:
![Environment Demo](/demos/001-sumorobot-with-box.gif)

Trained agent demo:
![Trained agent Demo](/demos/002-sumorobot-with-box-trained.gif)