# import matplotlib.pyplot as plt
from rllib.util import rollout_policy
from rllib.environment.systems import InvertedPendulum, GaussianSystem
from rllib.environment import SystemEnvironment
from rllib.policy import FelixPolicy
import numpy as np
import torch.optim

NUM_EPISODES = 10
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

system = InvertedPendulum(mass=0.1, length=0.5, friction=0.)
# system = system.linearize()

system = GaussianSystem(system, transition_noise_scale=0, measurement_noise_scale=0)


def initial_state():
    return np.array([np.deg2rad(20), 0.])


def termination(state):
    return np.abs(state[..., 0]) >= np.deg2rad(45)


def reward_function(state, action):
    return torch.exp(-0.5 / (0.2 ** 2) * torch.tensor(state[..., 0]) ** 2)


environment = SystemEnvironment(system, initial_state, reward=reward_function,
                                termination=termination, max_steps=50)

policy = FelixPolicy(environment.dim_state, environment.dim_action, temperature=0.)


trajectories = rollout_policy(environment, policy, NUM_EPISODES)
for trajectory in trajectories:
    # dataset.append(trajectory)
    print(f'length: {len(trajectory)}')
