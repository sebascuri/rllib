import numpy as np
import torch.distributions


from experiments.gpucrl_inverted_pendulum.util import PendulumModel
from experiments.gpucrl_inverted_pendulum.util import solve_mpc

torch.manual_seed(0)
np.random.seed(0)

dynamic_model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)

action_cost_ratio = 0.2

num_iter = 100
num_sim_steps = 200
batch_size = 128
refresh_interval = 1
num_trajectories = 16
num_episodes = 1
num_action_samples = 8
epsilon, epsilon_mean, epsilon_var = 0.1, 1., 0.001
lr = 5e-4

solve_mpc(dynamic_model, action_cost_ratio=action_cost_ratio,
          num_iter=num_iter, num_sim_steps=num_sim_steps, batch_size=batch_size,
          refresh_interval=refresh_interval, num_trajectories=num_trajectories,
          num_action_samples=num_action_samples, num_episodes=num_episodes,
          epsilon=epsilon, epsilon_mean=epsilon_mean, epsilon_var=epsilon_var,
          lr=lr)
