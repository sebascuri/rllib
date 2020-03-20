import math

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Uniform
import torch.optim as optim
from gpytorch.distributions import Delta

from rllib.algorithms.control.mppo import MBMPPO, train_mppo
from rllib.dataset.dataset import TrajectoryDataset
from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import MeanFunction, StateActionNormalizer, ActionClipper
from rllib.dataset.utilities import stack_list_of_tuples, bootstrap_trajectory
from rllib.environment.system_environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.model.gp_model import ExactGPModel
from rllib.model.pendulum_model import PendulumModel
from rllib.model.unscaled_model import UnscaledModel
from rllib.model.nn_model import NNModel
from rllib.policy import NNPolicy
from rllib.reward.pendulum_reward import PendulumReward
from rllib.util.collect_data import collect_model_transitions

from rllib.util.utilities import tensor_to_distribution
from rllib.util.plotting import plot_learning_losses, plot_values_and_policy
from rllib.util.rollout import rollout_model, rollout_policy
from rllib.util.neural_networks import DeterministicEnsemble
from rllib.util.neural_networks.utilities import freeze_parameters
from rllib.value_function import NNValueFunction

from experiments.util import train_model

torch.manual_seed(0)
np.random.seed(0)


class StateTransform(nn.Module):
    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        angle, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat((torch.cos(angle), torch.sin(angle), angular_velocity),
                            dim=-1)
        return states_


# %% Collect Data.
ensemble_size = 5
num_data = 500
reward_model = PendulumReward()
dynamic_model = PendulumModel(mass=0.3, length=0.5, friction=0.005)

transitions = collect_model_transitions(
    Uniform(torch.tensor([-2 * np.pi, -12]), torch.tensor([2 * np.pi, 12])),
    Uniform(torch.tensor([-1.]), torch.tensor([1.])),
    dynamic_model, reward_model, num_data)

bootstraps = bootstrap_trajectory(transitions, ensemble_size)
dataset = [stack_list_of_tuples(t) for t in bootstraps]


nn_model = [NNModel(2, 1, layers=[64], input_transform=StateTransform(),
                    deterministic=True) for _ in range(ensemble_size)]


class EnsembleModel(nn.Module):
    def __init__(self, in_dim, out_dim, layers, input_transform=None, num_heads=5):
        super().__init__()
        self.input_transform = input_transform
        if hasattr(input_transform, 'extra_dim'):
            in_dim += getattr(input_transform, 'extra_dim')
        self.nn = DeterministicEnsemble(in_dim, out_dim, layers=layers, non_linearity='ReLU',
                                        num_heads=num_heads)

    def forward(self, state, action):
        if self.input_transform is not None:
            expanded_state = self.input_transform(state)
        else:
            expanded_state = state

        state_action = torch.cat((expanded_state, action), dim=-1)
        next_state = self.nn(state_action)
        return state + next_state[0], next_state[1]

    @torch.jit.export
    def select_head(self, head_ptr):
        self.nn.select_head(head_ptr)


class Ensemble(nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.head = -1

    def forward(self, state, action):
        mean_ = []
        cov_ = []
        i = 0
        for m in self.models:
            mean_i, cov_i = m(state, action)
            mean_.append(mean_i)
            cov_.append(cov_i)
            if i == self.head:
                return mean_i, cov_i
            i += 1
        mean = torch.stack(mean_, dim=-1)
        mu = torch.mean(mean, dim=-1, keepdim=True)

        sigma = (mean - mu) @ (mean - mu).transpose(-2, -1)
        cov = torch.mean(torch.stack(cov_, dim=-1))
        if not torch.all(cov == 0):
            sigma = sigma + cov

        return mu.squeeze(-1), sigma

    @torch.jit.export
    def select_head(self, i: int):
        self.head = i


# ensemble = Ensemble(nn_model)
ensemble = EnsembleModel(3, 2, layers=[64], input_transform=StateTransform(), num_heads=ensemble_size)
ensemble = torch.jit.script(ensemble)

optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.01)
num_iter = 50
grad_steps = 3
for _ in range(num_iter):
    for i, data in enumerate(dataset):
        ensemble.nn.select_head(i)
        train_model(ensemble, data, grad_steps, optimizer)

for i in range(6):
    ensemble.nn.select_head(i)
    print(ensemble(dataset[0].state[0], dataset[0].action[0]))
# class MeanModel(nn.Module):
#     def forward(self, state, action):
#         """Pass the mean model."""
#         return state
#
# transformations = [
#     ActionClipper(-1, 1),
#     # MeanFunction(MeanModel()),
#     # StateActionNormalizer()
# ]
#
# dataset = TrajectoryDataset(sequence_length=1, transformations=transformations)
# dataset.append(transitions)
# data = dataset.all_data

# %% Train Model
num_iter = 200

# model = ExactGPModel(data.state, data.action, data.next_state)
# model.eval()
# model(torch.randn(1, 2), torch.randn(1, 1))
# model.train()
#
# # Use the adam optimizer
# optimizer = torch.optim.Adam([
#     {'params': model.parameters()},  # Includes GaussianLikelihood parameters
# ], lr=0.1)
#
# # "Loss" for GPs - the marginal log likelihood
# mlls = [gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
#         for likelihood, gp in zip(model.likelihood, model.gp)]
# for i in range(num_iter):
#     optimizer.zero_grad()
#     loss = torch.tensor(0.)
#     outputs = model(data.state, data.action)
#     for output_mean, output_cov, mll, next_state in zip(outputs[0], outputs[1], mlls, data.next_state.T):
#         output = tensor_to_distribution((output_mean, output_cov))
#         loss -= mll(output, next_state)
#     loss.backward()
#
#     optimizer.step()
#     print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iter, loss.item()))
#
# model.eval()
# outputs = model(data.state, data.action)
# l2 = torch.tensor(0.)
# for output_mean, output_cov, mll, next_state in zip(outputs[0], outputs[1], mlls,
#                                                     data.next_state.T):
#     l2 += torch.mean((output_mean - next_state) ** 2)
# print(l2)
#

#
# %% Define Policy, Value function, Model, Initial Distribution, Optimizer.
# dynamic_model = UnscaledModel(model, dataset.transformations)
# with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
#     dynamic_model.eval()
#     state, action = torch.randn(5, 20, 2), torch.randn(5, 20, 1)
#     pred = dynamic_model(state, action)
#     dynamic_model = torch.jit.trace(dynamic_model, (state, action))
dynamic_model = ensemble
freeze_parameters(dynamic_model)
value_function = NNValueFunction(dim_state=2, layers=[64, 64], biased_head=False,
                                 input_transform=StateTransform())

policy = NNPolicy(dim_state=2, dim_action=1, layers=[64, 64], biased_head=False,
                  squashed_output=True, input_transform=StateTransform())

value_function = torch.jit.script(value_function)
init_distribution = torch.distributions.Uniform(torch.tensor([-np.pi, -0.05]),
                                                torch.tensor([np.pi, 0.05]))

# Initialize MPPO and optimizer.
mppo = MBMPPO(dynamic_model, reward_model, policy, value_function,
              epsilon=0.1, epsilon_mean=0.01, epsilon_var=0.00, gamma=0.99,
              num_action_samples=15)

optimizer = optim.Adam([p for p in mppo.parameters() if p.requires_grad], lr=5e-4)

# %%  Train Controller
num_iter = 100
num_simulation_steps = 400
batch_size = 100
refresh_interval = 2
num_inner_iterations = 30
num_trajectories = math.ceil(num_inner_iterations * 100 / num_simulation_steps)
num_subsample = 1

value_losses, policy_losses, policy_returns, eta_parameters = train_mppo(
    mppo, init_distribution, optimizer,
    num_iter=num_iter, num_trajectories=num_trajectories,
    num_simulation_steps=num_simulation_steps, refresh_interval=refresh_interval,
    batch_size=batch_size, num_subsample=num_subsample)

plt.plot(refresh_interval * np.arange(len(policy_returns)), policy_returns)
plt.xlabel('Iteration')
plt.ylabel('Cumulative reward')
plt.show()

plot_learning_losses(policy_losses, value_losses, horizon=20)
plt.show()


# %% Test controller on Model.
test_state = torch.tensor(np.array([np.pi, 0.]), dtype=torch.get_default_dtype())

with torch.no_grad():
    trajectory = rollout_model(mppo.dynamical_model, mppo.reward_model,
                               lambda x: (policy(x)[0], torch.zeros(1)),
                               initial_state=test_state.unsqueeze(0),
                               max_steps=400)

    trajectory = Observation(*stack_list_of_tuples(trajectory))

states = trajectory.state[0]
rewards = trajectory.reward
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))

plt.sca(ax1)
plt.plot(states[:, 0], states[:, 1], 'x')
plt.plot(states[-1, 0], states[-1, 1], 'x')
plt.xlabel('Angle [rad]')
plt.ylabel('Angular velocity [rad/s]')

plt.sca(ax2)
plt.plot(rewards)
plt.xlabel('Time step')
plt.ylabel('Instantaneous reward')
plt.show()
print(f'Cumulative reward: {torch.sum(rewards):.2f}')

bounds = [(-2 * np.pi, 2 * np.pi), (-12, 12)]
ax_value, ax_policy = plot_values_and_policy(value_function, policy, bounds, [200, 200])
ax_value.plot(states[:, 0], states[:, 1], color='C1')
ax_value.plot(states[-1, 0], states[-1, 1], 'x', color='C1')
plt.show()

# %% Test controller on Environment.
environment = SystemEnvironment(InvertedPendulum(mass=0.3, length=0.5, friction=0.005,
                                                 step_size=1 / 80))
environment.state = test_state.numpy()
environment.initial_state = lambda: test_state.numpy()
rollout_policy(environment, lambda x: (policy(x)[0], torch.zeros(1)), max_steps=400, render=True
               )
