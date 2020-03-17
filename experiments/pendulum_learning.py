from rllib.model.pendulum_model import PendulumModel
from rllib.reward.pendulum_reward import PendulumReward
from rllib.dataset.transforms import MeanFunction, StateActionNormalizer, ActionClipper
from rllib.dataset.dataset import TrajectoryDataset
from rllib.dataset.datatypes import Observation
from rllib.model.gp_model import ExactGPModel
from rllib.algorithms.control.mppo import MBMPPO, train_mppo
from rllib.environment.system_environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum

import math
from rllib.util.rollout import rollout_model, rollout_policy
from rllib.dataset.utilities import stack_list_of_tuples
import gpytorch
from gpytorch.distributions import Delta
from rllib.model.unscaled_model import UnscaledModel
import torch.optim as optim
from rllib.value_function import NNValueFunction
from rllib.policy import NNPolicy
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
from rllib.util.plotting import plot_learning_losses, plot_values_and_policy
torch.manual_seed(0)


# %% Collect Data.
num_data = 200
reward_model = PendulumReward()
dynamic_model = PendulumModel(mass=0.3, length=0.5, friction=0.005)

state_action_dist = torch.distributions.Uniform(torch.tensor([-2 * np.pi, -12, -1]),
                                                torch.tensor([2 * np.pi, 12, 1]))
state_actions = state_action_dist.sample((num_data,))
states, actions = state_actions[:, :2], state_actions[:, 2:]
next_states = dynamic_model(states, actions).sample()
rewards = reward_model(states, actions).mean

transformations = [
    ActionClipper(-1, 1), MeanFunction(lambda s, a: s),
    # StateActionNormalizer()
]

trajectory = []
for state, action, reward, next_state in zip(states, actions, rewards, next_states):
    trajectory.append(
        Observation(state, action, reward, next_state).to_torch())

dataset = TrajectoryDataset(sequence_length=1, transformations=transformations)
dataset.append(trajectory)
data = dataset.all_data

# %% Train GP Model
num_iter = 50

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

    optimizer.step()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iter, loss.item()))


# %% Define Policy, Value function, Model, Initial Distribution, Optimizer.

def state_transform(states_):
    """Transform state before applying function approximation."""
    angle, angular_velocity = torch.split(states_, 1, dim=-1)
    states_ = torch.cat((torch.cos(angle), torch.sin(angle), angular_velocity),
                        dim=-1)
    return states_


dynamic_model = UnscaledModel(model, dataset.transformations)
value_function = NNValueFunction(dim_state=3, layers=[64, 64], biased_head=False,
                                 input_transform=state_transform)
policy = NNPolicy(dim_state=3, dim_action=1, layers=[64, 64], biased_head=False,
                  squashed_output=True, input_transform=state_transform)

init_distribution = torch.distributions.Uniform(torch.tensor([-np.pi, -0.05]),
                                                torch.tensor([np.pi, 0.05]))

# Initialize MPPO and optimizer.
mppo = MBMPPO(dynamic_model, reward_model, policy, value_function,
              epsilon=0.1, epsilon_mean=0.01, epsilon_var=0.00, gamma=0.99,
              num_action_samples=15)

optimizer = optim.Adam(mppo.parameters(), lr=5e-4)

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
plt.ion()
plt.show()

plot_learning_losses(policy_losses, value_losses, horizon=20)
plt.ion()
plt.show()


# %% Test controller on Model.
test_state = np.array([np.pi, 0.])

with torch.no_grad():
    trajectory = rollout_model(mppo.dynamical_model, mppo.reward_model,
                               lambda x: Delta(policy(x).mean),
                               initial_state=torch.tensor(test_state).float(),
                               max_steps=400)

    trajectory = Observation(*stack_list_of_tuples(trajectory))

states = trajectory.state
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
plt.ion()
plt.show()
print(f'Cumulative reward: {np.sum(rewards)}')

bounds = [(-2 * np.pi, 2 * np.pi), (-12, 12)]
ax_value, ax_policy = plot_values_and_policy(value_function, policy, bounds, [200, 200])
ax_value.plot(states[:, 0], states[:, 1], color='C1')
ax_value.plot(states[-1, 0], states[-1, 1], 'x', color='C1')
plt.ion()
plt.show()

# %% Test controller on Environment.
environment = SystemEnvironment(InvertedPendulum(mass=0.3, length=0.5, friction=0.005,
                                                 step_size=1 / 80))
environment.state = test_state
environment.initial_state = lambda: test_state
rollout_policy(environment, lambda x: Delta(policy(x).mean), max_steps=400, render=True
               )
