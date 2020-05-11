
import numpy as np
import torch.distributions

from exps.gpucrl.inverted_pendulum.util import PendulumModel
from exps.gpucrl.inverted_pendulum.util import solve_mpc

torch.manual_seed(0)
np.random.seed(0)

dynamical_model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)

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

returns = solve_mpc(
    dynamical_model, action_cost=action_cost,
    num_iter=num_iter, num_sim_steps=num_sim_steps, batch_size=batch_size,
    num_gradient_steps=num_gradient_steps, num_trajectories=num_trajectories,
    num_action_samples=num_action_samples, num_episodes=num_episodes,
    epsilon=epsilon, epsilon_mean=epsilon_mean, epsilon_var=epsilon_var,
    eta=eta, eta_mean=eta_mean, eta_var=eta_var,
    lr=lr)

