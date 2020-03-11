from rllib.model.pendulum_model import PendulumModel
from rllib.reward.pendulum_reward import PendulumReward
from rllib.dataset.transforms import MeanFunction, StateActionNormalizer, ActionClipper
from rllib.dataset.dataset import TrajectoryDataset
from rllib.dataset.datatypes import Observation
from rllib.model.gp_model import ExactGPModel

import torch
import torch.distributions
import numpy as np

torch.manual_seed(0)


# %% Define Reward Function
def state_transform(states):
    angle, angular_velocity = torch.split(states, 1, dim=-1)
    states = torch.cat((torch.cos(angle), torch.sin(angle), angular_velocity),
                       dim=-1)
    return states


reward_model = PendulumReward()

# %% Define Policy and Value functions.
dynamic_model = PendulumModel(mass=0.3, length=0.5, friction=0.005)

state_action_dist = torch.distributions.Uniform(torch.tensor([-2 * np.pi, -12, -1]),
                                                torch.tensor([2 * np.pi, 12, 1]))
state_actions = state_action_dist.sample((300,))
states, actions = state_actions[:, :2], state_actions[:, 2:]
next_states = dynamic_model(states, actions).sample()
rewards = reward_model(states, actions).mean

transformations = [
    ActionClipper(-1, 1), MeanFunction(lambda state, action: state),
    StateActionNormalizer()
]

trajectory = []
for state, action, reward, next_state in zip(states, actions, rewards, next_states):
    trajectory.append(
        Observation(state, action, reward, next_state).to_torch())

dataset = TrajectoryDataset(sequence_length=1, transformations=transformations)
dataset.append(trajectory[:50])
dataset.append(trajectory[50:])

data = dataset.all_data

# %% Optimize Model
import gpytorch
num_iter = 20

model = ExactGPModel(data.state, data.action, data.next_state)
model.eval()
model(torch.randn(1, 2), torch.randn(1, 1))
model.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model.gp)
for i in range(num_iter):
    optimizer.zero_grad()
    output = model(data.state, data.action)
    loss = -mll(output, data.next_state)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iter, loss.item()))
    optimizer.step()

# %% Evaluate Unscaled Model
from rllib.model.unscaled_model import UnscaledModel
ss = UnscaledModel(model, dataset.transformations)

state_actions = state_action_dist.sample((300,))
states, actions = state_actions[:, :2], state_actions[:, 2:]
next_states = dynamic_model(states, actions).sample()

pred = ss(states, actions)