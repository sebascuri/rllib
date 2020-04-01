import math

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Uniform
import torch.optim as optim

from rllib.agent.mbmppo_agent import MBMPPOAgent
from rllib.algorithms.mppo import MBMPPO
from rllib.dataset.datatypes import Observation
from rllib.dataset.transforms import MeanFunction, StateActionNormalizer, ActionClipper, \
    DeltaState
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.environment.system_environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.model.gp_model import ExactGPModel
from rllib.model.nn_model import NNModel
from rllib.model.pendulum_model import PendulumModel

from rllib.model.derived_model import TransformedModel, OptimisticModel, ExpectedModel
from rllib.model.ensemble_model import EnsembleModel
from rllib.policy import NNPolicy
from rllib.reward.pendulum_reward import PendulumReward

from rllib.util.plotting import plot_learning_losses, plot_values_and_policy, pendulum_gp_inputs_and_trajectory
from rllib.util.rollout import rollout_model, rollout_policy
from rllib.util.training import train_agent, evaluate_agent
from rllib.value_function import NNValueFunction

# %% Define Experiment Parameters.
hparams = {'seed': 0,
           'gamma': 0.99,
           'horizon': 400,
           'train_episodes': 30,
           'test_episodes': 1,
           'action_cost_ratio': 0,
           'optimistic': True,
           'learn_model': True,
           'exploratory_initial_distribution': False,
           'beta': 1.0,
           'max_memory': 1000,
           'batch_size': 100,
           'num_model_iter': 0,
           'num_mppo_iter': 30,
           'num_simulation_steps': 400,
           'state_refresh_interval': 2,
           'num_simulation_trajectories': 8,
           }

# model_params = {'kind': 'Ensemble', 'heads': 5, 'layers': [64], 'non_linearity': 'ReLU',
#                 'biased_head': True, 'deterministic': True}
# model_opt_params = {'lr': 1e-4, 'weight_decay': 0}

model_params = {'kind': 'ExactGP', 'max_num_points': 150}
model_opt_params = {'lr': 1e-1, 'weight_decay': 0}

policy_params = {'layers': [64, 64], 'non_linearity': 'ReLU', 'squashed_output': True,
                 'biased_head': False, 'deterministic': False}
policy_opt_params = {'lr': 5e-4, 'weight_decay': 0}
mppo_params = {'epsilon': 0.1, 'epsilon_mean': 0.01, 'epsilon_var': 0.00,
               'num_action_samples': 15}
vf_params = {'layers': [64, 64], 'non_linearity': 'ReLU',
             'biased_head': False}

torch.manual_seed(hparams['seed'])
np.random.seed(hparams['seed'])


class StateTransform(nn.Module):
    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        angle, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat((torch.cos(angle), torch.sin(angle), angular_velocity),
                            dim=-1)
        return states_


def termination(state, action, next_state=None):
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    return torch.any(torch.abs(state) > 15) or torch.any(torch.abs(action) > 15)


transformations = [
    ActionClipper(-1, 1),
    MeanFunction(DeltaState()),
    # StateActionNormalizer()
]

# %% Define Environment.
reward_model = PendulumReward(action_cost_ratio=hparams['action_cost_ratio'])
if hparams['exploratory_initial_distribution']:
    initial_distribution = torch.distributions.Uniform(
        torch.tensor([-np.pi, -0.05]),
        torch.tensor([np.pi, 0.05])
    )
else:
    initial_distribution = torch.distributions.Uniform(
        torch.tensor([np.pi, 0.0]),
        torch.tensor([np.pi, +0.0])
    )

environment = SystemEnvironment(InvertedPendulum(mass=0.3, length=0.5, friction=0.005,
                                                 step_size=1 / 80),
                                reward=reward_model,
                                initial_state=initial_distribution.sample,
                                termination=termination)

# %% Define Model
if hparams['learn_model']:
    if model_params['kind'] == 'ExactGP':
        model = ExactGPModel(torch.tensor([[np.pi, 0.0]]), torch.tensor([[0.0]]),
                             torch.tensor([[0.0, 0.0]]),
                             input_transform=StateTransform(),
                             max_num_points=model_params['max_num_points'])
        model.gp[0].covar_module.outputscale = torch.tensor(0.0042)
        model.gp[1].covar_module.outputscale = torch.tensor(0.56)
        model.gp[0].covar_module.base_kernel.lengthscale = torch.tensor([[8.3]])
        model.gp[1].covar_module.base_kernel.lengthscale = torch.tensor([[9.0]])

        model.likelihood[0].noise = torch.tensor([1e-4])
        model.likelihood[1].noise = torch.tensor([1e-4])

    elif model_params['kind'] == 'Ensemble':
        model = EnsembleModel(
            environment.dim_state, environment.dim_action,
            num_heads=model_params['heads'], layers=model_params['layers'],
            biased_head=model_params['biased_head'],
            non_linearity=model_params['non_linearity'],
            input_transform=StateTransform(),
            deterministic=model_params['deterministic'])
    hparams.update({f"model_{key}": value for key, value in model_params.items()})
else:
    model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)
    hparams['num_model_iter'] = 0
    transformations = []

hparams.update({"model": model.__class__.__name__})

if hparams['optimistic']:
    dynamic_model = OptimisticModel(model, transformations, beta=hparams['beta'])
    dim_policy_action = environment.dim_action + environment.dim_state
else:
    dynamic_model = TransformedModel(model, transformations)
    dim_policy_action = environment.dim_action
hparams.update({"dynamic_model": dynamic_model.__class__.__name__})

# with gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
#     dynamic_model.eval()
#     s, a = torch.randn(15, 200, 2), torch.randn(15, 200, dim_policy_action)
#     dynamic_model(s, a)
#     dynamic_model = torch.jit.trace(dynamic_model, (s, a))

if hparams['learn_model']:
    model_optimizer = optim.Adam(dynamic_model.parameters(), lr=model_opt_params['lr'],
                                 weight_decay=model_opt_params['weight_decay'])
    hparams.update({"model-opt": model_optimizer.__class__.__name__})
    hparams.update({f"model-opt-{key}": val if not isinstance(val, tuple) else list(val)
                    for key, val in model_optimizer.defaults.items()
                    })
else:
    model_optimizer = None

# %% Define Policy
policy = NNPolicy(
    dim_state=environment.dim_state, dim_action=dim_policy_action,
    layers=policy_params['layers'], biased_head=policy_params['biased_head'],
    non_linearity=policy_params['non_linearity'],
    squashed_output=policy_params['squashed_output'],
    input_transform=StateTransform(), deterministic=policy_params['deterministic'])
hparams.update({"policy": policy.__class__.__name__})
hparams.update({f"policy_{key}": value for key, value in policy_params.items()})

# %% Define Value Function
value_function = NNValueFunction(dim_state=environment.dim_state,
                                 layers=vf_params['layers'],
                                 biased_head=vf_params['biased_head'],
                                 input_transform=StateTransform())
hparams.update({"value_function": value_function.__class__.__name__})
hparams.update({f"value_function_{key}": value for key, value in vf_params.items()})

# %% Define MPPO
mppo = MBMPPO(dynamic_model, reward_model, policy, value_function,
              epsilon=mppo_params['epsilon'], epsilon_mean=mppo_params['epsilon_mean'],
              epsilon_var=mppo_params['epsilon_var'], gamma=hparams['gamma'],
              num_action_samples=mppo_params['num_action_samples'],
              termination=termination)

mppo_optimizer = optim.Adam([p for name, p in mppo.named_parameters()
                             if 'model' not in name], lr=policy_opt_params['lr'],
                            weight_decay=model_opt_params['weight_decay'])
hparams.update(mppo_params)

hparams.update({
    "mppo-opt": mppo_optimizer.__class__.__name__,
})
hparams.update({f"mppo-opt-{key}": val if not isinstance(val, tuple) else list(val)
                for key, val in mppo_optimizer.defaults.items()
                })

# %% Define Agent
comment = model.__class__.__name__
comment += f"{'Optimistic' if hparams['optimistic'] else 'Expected'}"
comment += f"{'InitX' if hparams['exploratory_initial_distribution'] else 'InitF'}"

agent = MBMPPOAgent(environment.name, mppo, model_optimizer, mppo_optimizer,
                    transformations=transformations,
                    max_memory=hparams['max_memory'], batch_size=hparams['batch_size'],
                    num_model_iter=hparams['num_model_iter'],
                    num_mppo_iter=hparams['num_mppo_iter'],
                    state_refresh_interval=hparams['state_refresh_interval'],
                    num_simulation_trajectories=hparams['num_simulation_trajectories'],
                    num_simulation_steps=hparams['num_simulation_steps'],
                    gamma=hparams['gamma'], comment=comment)

# Train Agent
with gpytorch.settings.fast_computations(), gpytorch.settings.fast_pred_var(), \
      gpytorch.settings.fast_pred_samples():
    train_agent(agent, environment, num_episodes=hparams['train_episodes'],
                max_steps=hparams['horizon'], plot_flag=True, print_frequency=1,
                render=False, plot_callbacks=[pendulum_gp_inputs_and_trajectory]
                )
agent.logger.export_to_json(hparams)

# %% Test controller on Environment.
test_state = np.array([np.pi, 0.])
environment.state = test_state
environment.initial_state = lambda: test_state
evaluate_agent(agent, environment, num_episodes=hparams['test_episodes'],
               max_steps=hparams['horizon'], render=True)

returns = np.mean(agent.logger.get('environment_return')[-hparams['test_episodes']:])
agent.logger.writer.add_hparams(hparams, {"test/environment_returns": returns})

# %% Test controller on Sampled Model.
sampled_model = TransformedModel(model, transformations)
test_state = torch.tensor(test_state, dtype=torch.get_default_dtype())
with torch.no_grad():
    trajectory = rollout_model(sampled_model, reward_model,
                               lambda x: (agent.policy(x)[0], torch.zeros(1)),
                               initial_state=test_state.unsqueeze(0),
                               max_steps=hparams['horizon'])

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
print(f'Model Cumulative reward: {torch.sum(rewards):.2f}')

bounds = [(-2 * np.pi, 2 * np.pi), (-12, 12)]
ax_value, ax_policy = plot_values_and_policy(value_function, agent.policy, bounds,
                                             [200, 200])
ax_value.plot(states[:, 0], states[:, 1], color='C1')
ax_value.plot(states[-1, 0], states[-1, 1], 'x', color='C1')
plt.show()
