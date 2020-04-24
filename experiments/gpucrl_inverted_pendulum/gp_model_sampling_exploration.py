import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributions import Uniform

from rllib.dataset.experience_replay import BootstrapExperienceReplay
from rllib.dataset.transforms import MeanFunction, DeltaState, ActionClipper
from rllib.model.gp_model import ExactGPModel
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
dynamic_model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)

transitions = collect_model_transitions(
    Uniform(torch.tensor([-2 * np.pi, -12]), torch.tensor([2 * np.pi, 12])),
    Uniform(torch.tensor([-1.]), torch.tensor([1.])),
    dynamic_model, reward_model, num_data)

# %% Bootstrap into different trajectories.
transformations = [
    ActionClipper(-1, 1),
    MeanFunction(DeltaState()),
    # StateActionNormalizer()
]
dataset = BootstrapExperienceReplay(
    max_len=int(1e4), transformations=transformations, num_bootstraps=1)
for transition in transitions:
    dataset.append(transition)

data = dataset.all_data
split = 50
# dataset._ptr = split
train_loader = DataLoader(dataset, batch_size=split, shuffle=False)

# %% Train a Model
model = ExactGPModel(data.state[:split, 0], data.action[:split, 0],
                     data.next_state[:split, 0],
                     input_transform=StateTransform(), max_num_points=75)

model.eval()
mean, stddev = model(torch.randn(8, 5, 2), torch.randn(8, 5, 1))

optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)
print([(gp.output_scale, gp.length_scale) for gp in model.gp])
train_model(model, train_loader, optimizer, max_iter=100)
print([(gp.output_scale, gp.length_scale) for gp in model.gp])

# %% Add data and re-train
model.add_data(data.state[split:2 * split, 0], data.action[split:2 * split, 0],
               data.next_state[split:2 * split, 0])

train_model(model, train_loader, optimizer, max_iter=70)
print([(gp.output_scale, gp.length_scale) for gp in model.gp])

model.add_data(data.state[2 * split:, 0], data.action[2 * split:, 0],
               data.next_state[2 * split:, 0])

# %% Define dynamical model.
model.gp[0].output_scale = torch.tensor(1.)
model.gp[0].length_scale = torch.tensor([[9.0]])
model.likelihood[0].noise = torch.tensor([1e-4])

model.gp[1].output_scale = torch.tensor(1.)
model.gp[1].length_scale = torch.tensor([[9.0]])
model.likelihood[1].noise = torch.tensor([1e-4])

for gp in model.gp:
    gp.prediction_strategy = None

model.eval()
dynamic_model = TransformedModel(model, transformations)

# %% SOLVE MPC
action_cost_ratio = 0.2

num_iter = 100
num_sim_steps = 400
batch_size = 100
refresh_interval = 2
num_trajectories = 8
num_episodes = 1
num_action_samples = 15
epsilon, epsilon_mean, epsilon_var = 0.1, 0.01, 0.


solve_mpc(dynamic_model, action_cost_ratio=action_cost_ratio,
          num_iter=num_iter, num_sim_steps=num_sim_steps, batch_size=batch_size,
          refresh_interval=refresh_interval, num_trajectories=num_trajectories,
          num_action_samples=num_action_samples, num_episodes=num_episodes,
          epsilon=epsilon, epsilon_mean=epsilon_mean, epsilon_var=epsilon_var)
