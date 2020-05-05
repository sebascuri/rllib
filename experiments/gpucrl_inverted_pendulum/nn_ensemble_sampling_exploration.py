import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributions import Uniform

from rllib.dataset.experience_replay import BootstrapExperienceReplay
from rllib.dataset.transforms import MeanFunction, DeltaState, ActionClipper
from rllib.model.ensemble_model import EnsembleModel
from rllib.model.derived_model import TransformedModel
from rllib.util.collect_data import collect_model_transitions
from rllib.util.training import train_model

from experiments.gpucrl_inverted_pendulum.util import StateTransform
from experiments.gpucrl_inverted_pendulum.util import PendulumReward, PendulumModel
from experiments.gpucrl_inverted_pendulum.util import solve_mpc

torch.manual_seed(0)
np.random.seed(0)

# %% Collect Data.
num_data = 200
reward_model = PendulumReward()
dynamical_model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)

transitions = collect_model_transitions(
    Uniform(torch.tensor([-2 * np.pi, -12]), torch.tensor([2 * np.pi, 12])),
    Uniform(torch.tensor([-1.]), torch.tensor([1.])),
    dynamical_model, reward_model, num_data)

# %% Bootstrap into different trajectories.
ensemble_size = 5
transformations = [
    ActionClipper(-1, 1),
    MeanFunction(DeltaState()),
    # StateActionNormalizer()
]
dataset = BootstrapExperienceReplay(max_len=int(1e4), transformations=transformations,
                                    num_bootstraps=ensemble_size)
for transition in transitions:
    dataset.append(transition)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# %% Train a Model
ensemble = EnsembleModel(2, 1, num_heads=ensemble_size, layers=[64], biased_head=True,
                         non_linearity='ReLU', input_transform=StateTransform(),
                         deterministic=True)

optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.01, weight_decay=1e-5)
train_model(ensemble, dataloader, max_iter=150, optimizer=optimizer)

# %% Define dynamical model.
ensemble.eval()
dynamical_model = TransformedModel(ensemble, transformations)

# %% SOLVE MPC
action_cost = 0.2

batch_size = 32
num_action_samples = 16
num_trajectories = 8
num_episodes = 1
epsilon, epsilon_mean, epsilon_var = None, None, None
eta, eta_mean, eta_var = 1., 1.7, 1.1
lr = 5e-4

num_iter = 100
num_sim_steps = 400
num_gradient_steps = 50

solve_mpc(dynamical_model,
          action_cost=action_cost,
          num_iter=num_iter, num_sim_steps=num_sim_steps, batch_size=batch_size,
          num_gradient_steps=num_gradient_steps, num_trajectories=num_trajectories,
          num_action_samples=num_action_samples, num_episodes=num_episodes,
          epsilon=epsilon, epsilon_mean=epsilon_mean, epsilon_var=epsilon_var,
          eta=eta, eta_mean=eta_mean, eta_var=eta_var,
          lr=lr)
