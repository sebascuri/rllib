from rllib.environment.systems import GridWorld
from rllib.environment import SystemEnvironment
import numpy as np

goal = np.array([3, 3])


def initial_state():
    return np.array([0, 0])


def reward(state, _):
    global goal
    return float(np.all(goal == state))


def termination(state):
    global goal
    return np.all(state == goal)


system = GridWorld(grid_size=np.array([4, 4]))
environment = SystemEnvironment(
    system, initial_state=initial_state, reward=reward, termination=termination,
    max_steps=100
)

state = environment.reset()
for _ in range(100):
    state = system.state
    action = environment.action_space.sample()
    next_state, reward, done, _ = environment.step(action)
    print(state, action, next_state, reward, done)
    if done:
        break
