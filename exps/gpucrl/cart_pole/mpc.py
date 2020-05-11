from dotmap import DotMap

import copy
import argparse

import gpytorch
import numpy as np
import torch.jit
from torch.distributions import Uniform
import torch.optim as optim

from rllib.agent.mpc_agent import MPCAgent
from rllib.algorithms.mpc import CEMShooting
from rllib.dataset.transforms import MeanFunction, ActionScaler, DeltaState, \
    AngleWrapper
from rllib.environment import GymEnvironment
from rllib.model.gp_model import ExactGPModel, RandomFeatureGPModel, SparseGPModel
from rllib.model.nn_model import NNModel
from rllib.model.environment_model import EnvironmentModel
from rllib.model.derived_model import TransformedModel, OptimisticModel
from rllib.model.ensemble_model import EnsembleModel
from rllib.reward.mujoco_rewards import CartPoleReward
from rllib.policy.mpc_policy import MPCPolicy
from rllib.util.training import train_agent, evaluate_agent
from rllib.value_function import NNValueFunction

from exps.gpucrl.cart_pole.plotters import plot_last_trajectory
from exps.gpucrl.cart_pole.util import termination, StateTransform

from exps.gpucrl.util import large_state_termination, get_mpc_agent
from exps.gpucrl.plotters import plot_last_trajectory
from exps.gpucrl.mpc_arguments import parser

parser.description = 'Run Swing-up Cart-Pole using Model-Based MPC.'
parser.set_defaults(action_cost=0.01, action_scale=3.,
                    environment_max_steps=50, train_episodes=0,
                    model_kind='DeterministicEnsemble', model_learn_num_iter=50,
                    model_opt_lr=1e-3, render_train=True)
args = parser.parse_args()
params = DotMap(vars(args))
torch.manual_seed(params.seed)
np.random.seed(params.seed)

# %% Define Helper modules
transformations = [
    ActionScaler(scale=3),
    MeanFunction(DeltaState()),
    AngleWrapper(indexes=[1]),
]

input_transform = StateTransform()

# %% Define Environment.
environment = GymEnvironment('MBRLCartPole-v0', action_cost=params.action_cost,
                             seed=params.seed)
reward_model = CartPoleReward(action_cost=params.action_cost)
exploratory_distribution = torch.distributions.Uniform(
    torch.tensor([-np.pi, -1.25, -0.05, -0.05]),
    torch.tensor([+np.pi, +1.25, +0.05, +0.05])
)

agent = get_mpc_agent(environment.name, environment.dim_state, environment.dim_action,
                      params, reward_model, transformations, input_transform,
                      initial_distribution=exploratory_distribution)

# %% Train Agent
with gpytorch.settings.fast_computations(), gpytorch.settings.fast_pred_var(), \
     gpytorch.settings.fast_pred_samples(), gpytorch.settings.memory_efficient():
    train_agent(agent, environment,
                num_episodes=params.train_episodes,
                max_steps=params.environment_max_steps,
                plot_flag=params.plot_train_results,
                print_frequency=params.print_frequency,
                render=params.render_train,
                plot_callbacks=[plot_last_trajectory]
                )
agent.logger.export_to_json(params.toDict())

# %% Test agent.
metrics = dict()
evaluate_agent(agent, environment, num_episodes=params.test_episodes,
               max_steps=params.environment_max_steps, render=params.render_test)

returns = np.mean(agent.logger.get('environment_return')[-params.test_episodes:])
metrics.update({"test/test_env_returns": returns})
returns = np.mean(agent.logger.get('environment_return')[:-params.test_episodes])
metrics.update({"test/train_env_returns": returns})

agent.logger.log_hparams(params.toDict(), metrics)
