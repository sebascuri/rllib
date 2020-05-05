import gpytorch
import numpy as np
import torch
from torch.distributions import Uniform
import torch.optim as optim

import argparse

from rllib.agent.mbmppo_agent import MBMPPOAgent
from rllib.algorithms.mppo import MBMPPO
from rllib.dataset.transforms import MeanFunction, ActionClipper, DeltaState
from rllib.environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.model.gp_model import ExactGPModel, RandomFeatureGPModel, SparseGPModel
from rllib.model.nn_model import NNModel
from rllib.model.derived_model import TransformedModel, OptimisticModel
from rllib.model.ensemble_model import EnsembleModel
from rllib.policy import NNPolicy

from rllib.util.training import train_agent, evaluate_agent
from rllib.value_function import NNValueFunction

from experiments.gpucrl_inverted_pendulum.plotters import plot_pendulum_trajectories, \
    plot_values_and_policy

from experiments.gpucrl_inverted_pendulum.util import StateTransform, termination
from experiments.gpucrl_inverted_pendulum.util import test_policy_on_model, \
    test_policy_on_environment
from experiments.gpucrl_inverted_pendulum.util import PendulumModel, PendulumReward

# %% Define and parse arguments.
parser = argparse.ArgumentParser(
    description='Run Swing-up Pendulum using Model-Based RL.')
parser.add_argument('--optimistic', action='store_true',
                    help='activate optimistic exploration.')
parser.add_argument('--exact-model', action='store_true', help='Use exact model.')
parser.add_argument('--seed', type=int, default=0,
                    help='initial random seed (default: 0).')
parser.add_argument('--model', type=str, default='ExactGP',
                    choices=['ExactGP', 'SparseGP', 'FeatureGP', 'NN', 'Ensemble'])
parser.add_argument('--sparse-approximation', type=str, default='DTC',
                    choices=['DTC', 'SOR', 'FITC'])
parser.add_argument('--feature-approximation', type=str, default='QFF',
                    choices=['QFF', 'RFF', 'OFF'])
parser.add_argument('--probabilistic-ensemble', action='store_true')

args = parser.parse_args()
print(args)

# %% Define Experiment Parameters.
hparams = {'seed': args.seed,
           'gamma': 0.99,
           'horizon': 400,
           'train_episodes': 0,  # 15,
           'test_episodes': 1,
           'action_cost': 0.2,
           'optimistic': args.optimistic,
           'learn_model': not args.exact_model,
           'beta': 1.0,
           'plan_horizon': 1,
           'plan_samples': 8,
           'plan_elite': 1,
           'max_memory': 10000,
           'batch_size': 32,
           'num_model_iter': 30,
           'num_mppo_iter': 50,  # 100,
           'num_simulation_steps': 400,
           'num_gradient_steps': 50,
           'num_simulation_trajectories': 8,
           'num_subsample': 1,
           }
torch.manual_seed(hparams['seed'])
np.random.seed(hparams['seed'])

# %% Define Helper modules
transformations = [ActionClipper(-1, 1), MeanFunction(DeltaState())]

# %% Define Environment.
reward_model = PendulumReward(action_cost=hparams['action_cost'])
initial_distribution = torch.distributions.Uniform(
    torch.tensor([np.pi, -0.0]),
    torch.tensor([np.pi, +0.0])
)

environment = SystemEnvironment(InvertedPendulum(mass=0.3, length=0.5, friction=0.005,
                                                 step_size=1 / 80),
                                reward=reward_model,
                                initial_state=initial_distribution.sample,
                                termination=termination)

# %% Define Model
model_opt_params = {'lr': 5e-4, 'weight_decay': 0}
if args.model == 'ExactGP':
    model_params = {'kind': 'ExactGP', 'max_num_points': int(1e10)}
    hparams.update({'num_model_iter': 0})

elif args.model == 'SparseGP':
    model_params = {'kind': 'SparseGP', 'max_num_points': int(1e10),
                    'approximation': args.sparse_approximation, 'q_bar': 2}
    hparams.update({'num_model_iter': 0})

elif args.model == 'FeatureGP':
    model_params = {'kind': 'RandomFeatureGP', 'max_num_points': int(1e10),
                    'approximation': args.feature_approximation, 'num_features': 625}
    hparams.update({'num_model_iter': 0})

elif args.model == 'NN':
    model_params = {'kind': 'NN', 'heads': 5, 'layers': [64],
                    'non_linearity': 'ReLU', 'biased_head': True,
                    'deterministic': False}
elif args.model == 'Ensemble':
    model_params = {'kind': 'Ensemble', 'heads': 5, 'layers': [64],
                    'non_linearity': 'ReLU', 'biased_head': True,
                    'deterministic': not args.probabilistic_ensemble}
else:
    raise NotImplementedError

if hparams['learn_model']:
    state, action = torch.tensor([[np.pi, 0.0]]), torch.tensor([[0.0]])
    next_state = torch.tensor([[0.0, 0.0]])
    if model_params['kind'] == 'ExactGP':
        model = ExactGPModel(state, action, next_state,
                             max_num_points=model_params['max_num_points'],
                             input_transform=StateTransform())
    elif model_params['kind'] == 'SparseGP':
        model = SparseGPModel(state, action, next_state,
                              approximation=model_params['approximation'],
                              q_bar=model_params['q_bar'],
                              max_num_points=model_params['max_num_points'],
                              input_transform=StateTransform()
                              )
    elif model_params['kind'] == 'RandomFeatureGP':
        model = RandomFeatureGPModel(state, action, next_state,
                                     num_features=model_params['num_features'],
                                     approximation=model_params['approximation'],
                                     max_num_points=model_params['max_num_points'],
                                     input_transform=StateTransform()
                                     )
    elif model_params['kind'] == 'Ensemble':
        model = EnsembleModel(
            environment.dim_state, environment.dim_action,
            num_heads=model_params['heads'], layers=model_params['layers'],
            biased_head=model_params['biased_head'],
            non_linearity=model_params['non_linearity'],
            input_transform=StateTransform(),
            deterministic=model_params['deterministic'])
    elif model_params['kind'] == 'NN':
        model = NNModel(
            environment.dim_state, environment.dim_action,
            layers=model_params['layers'],
            biased_head=model_params['biased_head'],
            non_linearity=model_params['non_linearity'],
            input_transform=StateTransform(),
            deterministic=model_params['deterministic']
        )
    else:
        raise NotImplementedError

    try:  # Select GP initial Model.
        for i in range(model.dim_state):
            model.gp[i].output_scale = torch.tensor(0.1)
            # model.gp[i].length_scale = torch.tensor([[4.0]])
            model.gp[i].length_scale = torch.tensor([[9.0]])
            model.likelihood[i].noise = torch.tensor([1e-4])
        # model.gp[0].output_scale = torch.tensor(0.0042)
        # model.gp[1].output_scale = torch.tensor(0.56)
    except AttributeError:
        pass

    hparams.update({f"model_{key}": value for key, value in model_params.items()})
else:
    model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)
    hparams['num_model_iter'] = 0
    transformations = []

hparams.update({"model": model.__class__.__name__})

if hparams['optimistic']:
    dynamical_model = OptimisticModel(model, transformations, beta=hparams['beta'])
    dim_policy_action = environment.dim_action + environment.dim_state
else:
    dynamical_model = TransformedModel(model, transformations)
    dim_policy_action = environment.dim_action
hparams.update({"dynamical_model": dynamical_model.__class__.__name__})
model_name = model.name
# with gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
#     dynamical_model.eval()
#     s, a = torch.randn(15, 200, 2), torch.randn(15, 200, dim_policy_action)
#     dynamical_model(s, a)
#     dynamical_model = torch.jit.trace(dynamical_model, (s, a))

if hparams['learn_model']:
    model_optimizer = optim.Adam(dynamical_model.parameters(), lr=model_opt_params['lr'],
                                 weight_decay=model_opt_params['weight_decay'])
    hparams.update({"model-opt": model_optimizer.__class__.__name__})
    hparams.update({f"model-opt-{key}": val if not isinstance(val, tuple) else list(val)
                    for key, val in model_optimizer.defaults.items()
                    })
else:
    model = torch.jit.script(dynamical_model)
    model_optimizer = None

# %% Define Policy
policy_params = {'layers': [64, 64], 'non_linearity': 'ReLU', 'squashed_output': True,
                 'biased_head': False, 'deterministic': False}

policy = NNPolicy(
    dim_state=environment.dim_state, dim_action=dim_policy_action,
    layers=policy_params['layers'], biased_head=policy_params['biased_head'],
    non_linearity=policy_params['non_linearity'],
    squashed_output=policy_params['squashed_output'],
    input_transform=StateTransform(), deterministic=policy_params['deterministic'])
hparams.update({"policy": policy.__class__.__name__})
hparams.update({f"policy_{key}": value for key, value in policy_params.items()})
policy = torch.jit.script(policy)

# %% Define Value Function
vf_params = {'layers': [64, 64], 'non_linearity': 'ReLU',
             'biased_head': False}

value_function = NNValueFunction(dim_state=environment.dim_state,
                                 layers=vf_params['layers'],
                                 biased_head=vf_params['biased_head'],
                                 input_transform=StateTransform())
hparams.update({"value_function": value_function.__class__.__name__})
hparams.update({f"value_function_{key}": value for key, value in vf_params.items()})
value_function = torch.jit.script(value_function)

# %% Define MPPO
mppo_params = {'eta': 1., 'eta_mean': 1.7, 'eta_var': 1.1, 'num_action_samples': 16}
mppo_opt_params = {'lr': 5e-4, 'weight_decay': 0}

mppo = MBMPPO(dynamical_model, reward_model, policy, value_function,
              eta=mppo_params.get('eta', None),
              eta_mean=mppo_params.get('eta_mean', None),
              eta_var=mppo_params.get('eta_var', None),
              epsilon=mppo_params.get('epsilon', None),
              epsilon_mean=mppo_params.get('epsilon_mean', None),
              epsilon_var=mppo_params.get('epsilon_var', None),
              gamma=hparams['gamma'],
              num_action_samples=mppo_params['num_action_samples'],
              termination=termination)

mppo_optimizer = optim.Adam([p for name, p in mppo.named_parameters()
                             if 'model' not in name], lr=mppo_opt_params['lr'],
                            weight_decay=mppo_opt_params['weight_decay'])
hparams.update(mppo_params)

hparams.update({
    "mppo-opt": mppo_optimizer.__class__.__name__,
})
hparams.update({f"mppo-opt-{key}": val if not isinstance(val, tuple) else list(val)
                for key, val in mppo_optimizer.defaults.items()
                })

# %% Define Agent
comment = model_name
comment += f"{' Optimistic' if hparams['optimistic'] else ' Expected'}"

agent = MBMPPOAgent(
    environment.name, mppo, model_optimizer, mppo_optimizer,
    initial_distribution=torch.distributions.Uniform(
        torch.tensor([-np.pi, -0.005]), torch.tensor([np.pi, +0.005])),
    plan_horizon=hparams['plan_horizon'], plan_samples=hparams['plan_samples'],
    plan_elite=hparams['plan_elite'],
    max_memory=hparams['max_memory'], batch_size=hparams['batch_size'],
    num_model_iter=hparams['num_model_iter'],
    num_mppo_iter=hparams['num_mppo_iter'],
    num_gradient_steps=hparams['num_gradient_steps'],
    num_subsample=hparams['num_subsample'],
    num_simulation_trajectories=hparams['num_simulation_trajectories'] // 2,
    num_distribution_trajectories=hparams['num_simulation_trajectories'] // 2,
    num_simulation_steps=hparams['num_simulation_steps'],
    gamma=hparams['gamma'], comment=comment)

print(agent)

# %% Train Agent
with gpytorch.settings.fast_computations(), gpytorch.settings.fast_pred_var(), \
     gpytorch.settings.fast_pred_samples(), gpytorch.settings.memory_efficient():
    train_agent(agent, environment, num_episodes=hparams['train_episodes'],
                max_steps=hparams['horizon'], plot_flag=True, print_frequency=1,
                render=False, plot_callbacks=[plot_pendulum_trajectories]
                )
agent.logger.export_to_json(hparams)

# %% Test agent.
metrics = dict()
test_state = torch.tensor(np.array([np.pi, 0.]), dtype=torch.get_default_dtype())

# environment.state = test_state.numpy()
# environment.initial_state = lambda: test_state.numpy()
# evaluate_agent(agent, environment, num_episodes=hparams['test_episodes'],
#                max_steps=hparams['horizon'], render=True)

# returns = np.mean(agent.logger.get('environment_return')[-hparams['test_episodes']:])
# metrics.update({"test/environment_returns": returns})

# Test Policy on Environment and Model.
returns, _ = test_policy_on_environment(environment, agent.policy, test_state)
metrics.update({"test/policy_environment": returns})

returns, _ = test_policy_on_model(dynamical_model, reward_model, agent.policy, test_state)
metrics.update({"test/policy_model": returns})

# Test Mean Policy on Environment and Model.
returns, _ = test_policy_on_environment(
    environment,
    lambda x: (agent.policy(x)[0][:agent.dynamical_model.base_model.dim_action],
               torch.zeros(1)), test_state, policy_str='Expected Policy')
metrics.update({"test/expected_policy_environment": returns})

returns, _ = test_policy_on_model(
    dynamical_model, reward_model,
    lambda x: (agent.policy(x)[0], torch.zeros(1)),
    test_state, policy_str='Expected Policy')
metrics.update({"test/expected_policy_model": returns})

# %% Test Policy on Sampled Model.
sampled_model = TransformedModel(model, transformations)
returns, _ = test_policy_on_model(sampled_model, reward_model, agent.policy, test_state)
metrics.update({"test/policy_sampled_model": returns})

returns, _ = test_policy_on_model(
    sampled_model, reward_model,
    lambda x: (agent.policy(x)[0][:sampled_model.dim_action],
               torch.zeros(1)), test_state, policy_str='Expected Policy')
metrics.update({"test/expected_policy_sampled_model": returns})

# plot_values_and_policy(mppo.value_function, mppo.policy, trajectory=trajectory,
#                        num_entries=[200, 200],
#                        bounds=[(-2 * np.pi, 2 * np.pi), (-12, 12)])

agent.logger.log_hparams(hparams, metrics)
