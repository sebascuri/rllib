from rllib.environment.systems import InvertedPendulum
from rllib.environment import SystemEnvironment
from rllib.policy import MLPPolicy
from rllib.value_function import NNValueFunction
import torch.nn as nn
import torch.optim as optim
from rllib.model import LinearModel
import numpy as np
import matplotlib.pyplot as plt

import rllib.model.utilities
from rllib.agent import DDPGAgent
from tqdm import tqdm
import torch
import rllib.util.neural_networks
import rllib.algorithms.control
from collections import OrderedDict


num_steps = 1
discount = 1.
batch_size = 40


system = InvertedPendulum(mass=0.1, length=0.5, friction=0.)
system = system.linearize()
q = np.eye(2)
r = 0.01 * np.eye(1)
gamma = 0.99

K, P = rllib.algorithms.control.dlqr(system.a, system.b, q, r, gamma=gamma)
K = torch.from_numpy(K.T).float()
P = torch.from_numpy(P).float()
q = torch.from_numpy(q).float()
r = torch.from_numpy(r).float()


def reward_function(state, action):
    state_cost = rllib.util.neural_networks.torch_quadratic(state, q)
    action_cost = rllib.util.neural_networks.torch_quadratic(action, r)
    return -(state_cost + action_cost).squeeze()


environment = SystemEnvironment(system, initial_state=None, termination=None,
                                reward=reward_function)


# class LinearModel(nn.Module):
#     """A linear Gaussian state space model."""
#
#     def __init__(self, a, b):
#         self.state_dim, self.action_dim = b.shape
#         super().__init__()
#
#         self.a = a.t()
#         self.b = b.t()
#
#     def forward(self, state, action):
#         return state @ self.a + action @ self.b
#

model = LinearModel(torch.from_numpy(system.a).float(),
                    torch.from_numpy(system.b).float())
# rllib.util.neural_networks.freeze_parameters(model)
model.a.requires_grad = False
model.b.requires_grad = False
# for param in model.parameters:
#     param.requires_grad = False

policy = MLPPolicy(dim_state=system.dim_state, dim_action=system.dim_action,
                   layers=[], biased_head=False, deterministic=True)  # Linear policy.
# policy = torch.nn.Linear(2, 1)

print(f'initial: {policy.policy.head.weight}')
value_function = NNValueFunction(dim_state=system.dim_state, layers=[64, 64, 64],
                                 biased_head=False)
# value_function = nn.Sequential(OrderedDict(
#     linear1=nn.Linear(model.dim_state, 64),
#     relu1=nn.ReLU(),
#     linear2=nn.Linear(64, 64),
#     relu2=nn.ReLU(),
#     linear3=nn.Linear(64, 1, bias=False)
# ))

# value_function = torch.jit.trace(value_function, torch.empty(3, 2))
# policy.policy = torch.jit.trace(policy.policy, torch.empty(3, 2))

loss_function = nn.MSELoss()
value_optimizer = optim.Adam(value_function.parameters, lr=5e-4)
policy_optimizer = optim.Adam(policy.parameters, lr=5e-4)

policy_losses = []
value_losses = []

for i in tqdm(range(10000)):
    value_optimizer.zero_grad()
    policy_optimizer.zero_grad()

    states = 0.5 * torch.randn(batch_size, 2)
    with rllib.util.neural_networks.disable_gradient(value_function.value_function):
        values = rllib.model.utilities.estimate_value(
            states=states,
            model=model,
            policy=policy,
            reward=reward_function,
            steps=0,
            gamma=gamma,
            bootstrap=value_function,
            num_samples=1)
    prediction = value_function(states)
    value_loss = loss_function(prediction, values)
    policy_loss = -values.mean()

    loss = policy_loss + value_loss
    loss.backward()
    policy_optimizer.step()
    value_optimizer.step()

    policy_losses.append(policy_loss.item())
    value_losses.append(value_loss.item())


horizon = 20
smoothing_weights = np.ones(horizon) / horizon
t = np.arange(len(policy_losses))
t_smooth = t[horizon // 2:-horizon // 2 + 1]

plt.plot(t, policy_losses)
plt.plot(t_smooth, np.convolve(policy_losses, smoothing_weights, 'valid'),
         label='smoothed')
plt.xlabel('Iteration')
plt.ylabel('Policy loss')
plt.legend()
plt.show()

plt.plot(t, value_losses)
plt.plot(t_smooth, np.convolve(value_losses, smoothing_weights, 'valid'),
         label='smoothed')
plt.xlabel('Iteration')
plt.ylabel('Value loss')
plt.legend()
plt.show()

print(f'optimal: {K}')
print(f'learned: {policy.policy.head.weight}')

bounds = [(-0.5, 0.5), (-0.5, 0.5)]
num_entries = [100, 100]
optimal_value = lambda x: rllib.util.neural_networks.torch_quadratic(x, matrix=-P)

states = rllib.util.linearly_spaced_combinations(bounds, num_entries)

values = optimal_value(torch.from_numpy(states).float())
img = rllib.util.plot_combinations_as_grid(plt.gca(), values.detach().numpy(),
                                           num_entries, bounds)
plt.colorbar(img)
plt.title('True value function')
plt.show()

values = value_function(torch.from_numpy(states).float())
img = rllib.util.plot_combinations_as_grid(plt.gca(), values.detach().numpy(),
                                           num_entries, bounds)
plt.colorbar(img)
plt.title('Learned value function')
plt.show()
