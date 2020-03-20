import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import rllib.algorithms.control
import rllib.util.neural_networks
from rllib.algorithms.dyna import dyna_rollout
from rllib.environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.model import LinearModel
from rllib.policy import NNPolicy
from rllib.reward.quadratic_reward import QuadraticReward
from rllib.value_function import NNValueFunction

num_steps = 1
discount = 1.
batch_size = 40

system = InvertedPendulum(mass=0.1, length=0.5, friction=0.)
system = system.linearize()
q = np.eye(2)
r = 0.01 * np.eye(1)
gamma = 0.99

K, P = rllib.algorithms.control.dlqr(system.a, system.b, q, r, gamma=gamma)
K = torch.from_numpy(K.T).type(torch.get_default_dtype())
P = torch.from_numpy(P).type(torch.get_default_dtype())

reward_model = QuadraticReward(torch.from_numpy(q).type(torch.get_default_dtype()),
                               torch.from_numpy(r).type(torch.get_default_dtype()))
environment = SystemEnvironment(system, initial_state=None, termination=None,
                                reward=lambda x, u: reward_model(x, u)[0])

model = LinearModel(system.a, system.b)

policy = NNPolicy(dim_state=system.dim_state, dim_action=system.dim_action,
                  layers=[], biased_head=False, deterministic=True)  # Linear policy.
print(f'initial: {policy.nn.head.weight}')

value_function = NNValueFunction(dim_state=system.dim_state, layers=[64, 64],
                                 biased_head=False)

policy = torch.jit.script(policy)
model = torch.jit.script(model)
value_function = torch.jit.script(value_function)

loss_function = nn.MSELoss()
value_optimizer = optim.Adam(value_function.parameters(), lr=5e-4)
policy_optimizer = optim.Adam(policy.parameters(), lr=5e-3)

policy_losses = []
value_losses = []
torch.autograd.set_detect_anomaly(True)
for i in tqdm(range(10000)):
    value_optimizer.zero_grad()
    policy_optimizer.zero_grad()

    states = 0.5 * torch.randn(batch_size, 2)
    with rllib.util.neural_networks.disable_gradient(value_function):
        dyna_return = dyna_rollout(state=states, model=model, policy=policy,
                                   reward=reward_model, steps=0, gamma=gamma,
                                   value_function=value_function, num_samples=5)
    prediction = value_function(states)
    value_loss = loss_function(prediction, dyna_return.q_target.mean(dim=0))
    policy_loss = -dyna_return.q_target.mean()

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
print(f'learned: {policy.nn.head.weight}')
# print(f'learned: {policy.weight}')

bounds = [(-0.5, 0.5), (-0.5, 0.5)]
num_entries = [100, 100]
optimal_value = lambda x: rllib.util.neural_networks.torch_quadratic(x, matrix=-P)

states = rllib.util.linearly_spaced_combinations(bounds, num_entries)

values = optimal_value(torch.from_numpy(states).type(torch.get_default_dtype()))
img = rllib.util.plot_combinations_as_grid(plt.gca(), values.detach().numpy(),
                                           num_entries, bounds)
plt.colorbar(img)
plt.title('True value function')
plt.show()

values = value_function(torch.from_numpy(states).type(torch.get_default_dtype()))
img = rllib.util.plot_combinations_as_grid(plt.gca(), values.detach().numpy(),
                                           num_entries, bounds)
plt.colorbar(img)
plt.title('Learned value function')
plt.show()
