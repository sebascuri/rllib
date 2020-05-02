"""Python Script Template."""
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from rllib.environment import GymEnvironment
from rllib.policy.mpc_policy import MPCPolicy
from rllib.model.environment_model import EnvironmentModel
from rllib.reward.environment_reward import EnvironmentReward
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.value_function import NNValueFunction
from rllib.util.utilities import tensor_to_distribution
from rllib.util.rollout import step

from rllib.algorithms.td import ModelBasedTDLearning


class EnvironmentTermination(nn.Module):
    def __init__(self, environment):
        super().__init__()
        self.environment = environment

    def forward(self, state, action, next_state=None):
        self.environment.state = state
        next_state, reward, done, _ = self.environment.step(action)
        return done


import copy

SEED = 0
MAX_ITER = 200
# ENVIRONMENT = 'VPendulum-v0'
ENVIRONMENT = 'VContinuous-CartPole-v0'

env = GymEnvironment(ENVIRONMENT, SEED)
env_model = copy.deepcopy(env)
env_model.reset()
dynamical_model = EnvironmentModel(env_model)
reward_model = EnvironmentReward(env_model)
termination = EnvironmentTermination(env_model)
GAMMA = 0.99
NUM_ITER = 20
horizon = 10
num_iter = 5
num_samples = 50
num_steps = horizon
solver = 'random_shooting'

memory = ExperienceReplay(max_len=2000, num_steps=1)

value_function = NNValueFunction(env.dim_state, layers=[64, 64])
optimizer = optim.Adam(value_function.parameters(), lr=1e-4)
policy = MPCPolicy(dynamical_model, reward_model, horizon, termination=termination,
                   terminal_reward=value_function,
                   num_iter=num_iter, num_samples=num_samples, solver=solver,
                   gamma=0.9)
value_learning = ModelBasedTDLearning(
    value_function, criterion=nn.MSELoss(reduction='none'), policy=policy,
    dynamical_model=dynamical_model, reward_model=reward_model,
    termination=termination, n_steps=num_steps, gamma=GAMMA)

total_td = []
total_rewards = []

for _ in range(10):
    state = env.reset()
    # env.state = np.array([0 + 0.1 * np.random.randn(), 0])
    # state = env.env._get_obs()

    done = False
    rewards = 0
    i = 0
    while not done:
        i += 1
        pi = tensor_to_distribution(policy(
            torch.tensor(state, dtype=torch.get_default_dtype()))
        )
        action = pi.sample().numpy()
        observation, state, done = step(env, state, action, pi, render=False)
        memory.append(observation)

        env.render()
        rewards += observation.reward.item()
        if i >= MAX_ITER:
            break
    total_rewards.append(rewards)
    print(rewards)
    for i in range(NUM_ITER):
        optimizer.zero_grad()
        observation, idx, weights = memory.get_batch(64)
        loss = value_learning(observation.state, observation.action, observation.reward,
                              observation.next_state, observation.done)

        loss.loss.mean().backward()
        optimizer.step()
        total_td.append(loss.td_error.abs().mean())

        value_learning.update()

    memory.reset()

import matplotlib.pyplot as plt

plt.plot(total_td)
plt.show()
