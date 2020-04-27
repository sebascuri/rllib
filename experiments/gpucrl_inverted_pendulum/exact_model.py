from itertools import product

import numpy as np
import torch.distributions
from tqdm import tqdm

from experiments.gpucrl_inverted_pendulum.util import PendulumModel
from experiments.gpucrl_inverted_pendulum.util import solve_mpc

torch.manual_seed(0)
np.random.seed(0)

dynamic_model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)

action_cost_ratio = 0.2

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
# (1.0, 1.9000000000000004, 0.5000000000000001)
# best_returns = -float('Inf')
# best_params = None
# for eta, eta_mean, eta_var in product([1.8, 2.0],
#                                       [1.1, 1.3, 1.5, 1.7, 1.9],
#                                       np.arange(0.3, 1.9, 0.2)):
#     results = {}
returns = solve_mpc(
    dynamic_model, action_cost_ratio=action_cost_ratio,
    num_iter=num_iter, num_sim_steps=num_sim_steps, batch_size=batch_size,
    num_gradient_steps=num_gradient_steps, num_trajectories=num_trajectories,
    num_action_samples=num_action_samples, num_episodes=num_episodes,
    epsilon=epsilon, epsilon_mean=epsilon_mean, epsilon_var=epsilon_var,
    eta=eta, eta_mean=eta_mean, eta_var=eta_var,
    lr=lr)
# params = (eta, eta_mean, eta_var)
# results[params] = returns
#
# if returns > best_returns:
#     best_returns = returns
#     best_params = params

# best = {k: v for k, v in filter(lambda item: item[1] > 200,
#                                 sorted(results.items(), key=lambda item: item[1]))}
#
# for i in range(3):
#     print(np.unique([k[i] for k in best.keys()], return_counts=True))
