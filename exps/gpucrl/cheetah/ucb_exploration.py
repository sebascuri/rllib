import gpytorch
import numpy as np
import torch.jit
from torch.distributions import Uniform
import torch.optim as optim

import argparse

from rllib.agent.mbmppo_agent import MBMPPOAgent
from rllib.algorithms.mppo import MBMPPO
from rllib.dataset.transforms import MeanFunction, ActionScaler, DeltaState
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.environment import GymEnvironment
from rllib.model.gp_model import ExactGPModel, RandomFeatureGPModel, SparseGPModel
from rllib.model.nn_model import NNModel
from rllib.model.derived_model import TransformedModel, OptimisticModel
from rllib.model.ensemble_model import EnsembleModel
from rllib.reward.mujoco_rewards import CartPoleReward
from rllib.policy import NNPolicy

from rllib.util.training import train_agent, evaluate_agent
from rllib.value_function import NNValueFunction

from experiments.gpucrl.cart_pole.plotters import plot_cartpole_trajectories
from experiments.gpucrl.cart_pole.util import termination

# %% Define and parse arguments.
parser = argparse.ArgumentParser(
    description='Run Swing-up CartPole using Model-Based RL.')
parser.add_argument('--optimistic', action='store_true',
                    help='activate optimistic exploration.')
parser.add_argument('--exact-model', action='store_true', help='Use exact model.')
parser.add_argument('--seed', type=int, default=0,
                    help='initial random seed (default: 0).')
parser.add_argument('--model-kind', type=str, default='ExactGP',
                    choices=['ExactGP', 'SparseGP', 'FeatureGP',
                             'ProbabilisticNN', 'DeterministicNN',
                             'ProbabilisticEnsemble', 'DeterministicEnsemble',
                             ])
parser.add_argument('--model-sparse-approximation', type=str, default='DTC',
                    choices=['DTC', 'SOR', 'FITC'])
parser.add_argument('--model-feature-approximation', type=str, default='QFF',
                    choices=['QFF', 'RFF', 'OFF'])

parser.add_argument('--action-cost', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--environment-max-steps', type=int, default=200)
parser.add_argument('--train-episodes', type=int, default=15)
parser.add_argument('--test-episodes', type=int, default=1)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--plan-horizon', type=int, default=40)
parser.add_argument('--plan-samples', type=int, default=100)
parser.add_argument('--plan-elite', type=int, default=5)
parser.add_argument('--max-memory', type=int, default=10000)

parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num-model-iter', type=int, default=0)
parser.add_argument('--num-mppo-iter', type=int, default=5)
parser.add_argument('--num-gradient-steps', type=int, default=50)

parser.add_argument('--sim-num-steps', type=int, default=200)
parser.add_argument('--sim-initial-states-num-trajectories', type=int, default=4)
parser.add_argument('--sim-initial-dist-num-trajectories', type=int, default=32)
parser.add_argument('--sim-num-subsample', type=int, default=1)

parser.add_argument('--model-opt-lr', type=float, default=1e-4)
parser.add_argument('--model-opt-weight-decay', type=float, default=0)
parser.add_argument('--model-max-num-points', type=int, default=int(1e10))
parser.add_argument('--model-sparse-q-bar', type=int, default=2)
parser.add_argument('--model-num-features', type=int, default=625)
parser.add_argument('--model-layers', type=list, default=[64, 64])
parser.add_argument('--model-non-linearity', type=str, default='tanh')
parser.add_argument('--model-unbiased-head', action='store_false')
parser.add_argument('--model-num-heads', type=int, default=5)

parser.add_argument('--policy-layers', type=list, default=[64, 64])
parser.add_argument('--policy-non-linearity', type=str, default='ReLU')
parser.add_argument('--policy-unbiased-head', action='store_false')
parser.add_argument('--policy-deterministic', action='store_true')

parser.add_argument('--value-function-layers', type=list, default=[64, 64])
parser.add_argument('--value-function-non-linearity', type=str, default='ReLU')
parser.add_argument('--value-function-unbiased-head', action='store_false')

parser.add_argument('--mppo-opt-lr', type=float, default=1e-5)
parser.add_argument('--mppo-opt-weight-decay', type=float, default=0)

parser.add_argument('--mppo-eta', type=float, default=.1)
parser.add_argument('--mppo-eta-mean', type=float, default=.5)
parser.add_argument('--mppo-eta-var', type=float, default=.5)
parser.add_argument('--mppo-num-action-samples', type=int, default=16)

parser.add_argument('--plot-train-results', action='store_true')
parser.add_argument('--render-train', action='store_true')
parser.add_argument('--render-test', action='store_true')
parser.add_argument('--print-frequency', type=int, default=1)

# plot_flag = False, print_frequency = 1,
# render = False,


args = parser.parse_args()

# %% Define Experiment Parameters.
hparams = vars(args)
hparams.update({'learn_model': not args.exact_model})

torch.manual_seed(hparams['seed'])
np.random.seed(hparams['seed'])

# %% Define Helper modules
transformations = [ActionScaler(scale=3), MeanFunction(DeltaState())]
input_transform = None
# %% Define Environment.
environment = GymEnvironment('MBRLCartPole-v0',
                             action_cost=hparams['action_cost'],
                             seed=hparams['seed'])

initial_distribution = torch.distributions.Normal(
    torch.zeros(environment.dim_state),
    torch.ones(environment.dim_state)
)
reward_model = CartPoleReward(action_cost=hparams['action_cost'])
# environment.state = initial_distribution.sample()
# %% Define Base Model
state = torch.zeros(1, environment.dim_state)
action = torch.zeros(1, environment.dim_action)
next_state = torch.zeros(1, environment.dim_state)
if hparams['learn_model']:
    if hparams['model_kind'] == 'ExactGP':
        model = ExactGPModel(state, action, next_state,
                             max_num_points=hparams['model_max_num_points'],
                             input_transform=input_transform)
    elif hparams['model_kind'] == 'SparseGP':
        model = SparseGPModel(state, action, next_state,
                              approximation=hparams['model_sparse_approximation'],
                              q_bar=hparams['model_sparse_q_bar'],
                              max_num_points=hparams['model_max_num_points'],
                              input_transform=input_transform
                              )
    elif hparams['model_kind'] == 'FeatureGP':
        model = RandomFeatureGPModel(
            state, action, next_state,
            num_features=hparams['model_num_features'],
            approximation=hparams['model_feature_approximation'],
            max_num_points=hparams['model_max_num_points'],
            input_transform=input_transform)
    elif hparams['model_kind'] in ['ProbabilisticEnsemble', 'DeterministicEnsemble']:
        model = EnsembleModel(
            environment.dim_state, environment.dim_action,
            num_heads=hparams['model_num_heads'], layers=hparams['model_layers'],
            biased_head=not hparams['model_unbiased_head'],
            non_linearity=hparams['model_non_linearity'],
            input_transform=input_transform,
            deterministic=hparams['model_kind'] == 'DeterministicEnsemble')
    elif hparams['model_kind'] in ['ProbabilisticNN', 'DeterministicNN']:
        model = NNModel(
            environment.dim_state, environment.dim_action,
            layers=hparams['model_layers'],
            biased_head=not hparams['model_unbiased_head'],
            non_linearity=hparams['model_non_linearity'],
            input_transform=input_transform,
            deterministic=hparams['model_kind'] == 'DeterministicNN'
        )
    else:
        raise NotImplementedError
    try:  # Select GP initial Model.
        for i in range(model.dim_state):
            model.gp[i].output_scale = torch.tensor(0.1)
            model.gp[i].length_scale = torch.tensor([[15.0]])
            model.likelihood[i].noise = torch.tensor([1e-4])
    except AttributeError:
        pass
else:
    model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)
    hparams['num_model_iter'] = 0
    # transformations = []

hparams.update({"model": model.__class__.__name__})

# %% Define Optimistic or Expected Model
if hparams['optimistic']:
    dynamical_model = OptimisticModel(model, transformations, beta=hparams['beta'])
    dim_policy_action = environment.dim_action + environment.dim_state
else:
    dynamical_model = TransformedModel(model, transformations)
    dim_policy_action = environment.dim_action

if hparams['learn_model']:
    model_optimizer = optim.Adam(dynamical_model.parameters(),
                                 lr=hparams['model_opt_lr'],
                                 weight_decay=hparams['model_opt_weight_decay'])
else:
    model = torch.jit.script(dynamical_model)
    model_optimizer = None

# %% Define Policy
policy = NNPolicy(
    dim_state=environment.dim_state, dim_action=dim_policy_action,
    layers=hparams['policy_layers'],
    biased_head=not hparams['policy_unbiased_head'],
    non_linearity=hparams['policy_non_linearity'],
    squashed_output=True,
    input_transform=input_transform,
    deterministic=hparams['policy_deterministic'])
hparams.update({"policy": policy.__class__.__name__})
policy = torch.jit.script(policy)

# %% Define Value Function
value_function = NNValueFunction(
    dim_state=environment.dim_state,
    layers=hparams['value_function_layers'],
    biased_head=not hparams['value_function_unbiased_head'],
    non_linearity=hparams['value_function_non_linearity'],
    input_transform=input_transform)
hparams.update({"value_function": value_function.__class__.__name__})
value_function = torch.jit.script(value_function)

# %% Define MPPO
mppo = MBMPPO(dynamical_model, reward_model, policy, value_function,
              eta=hparams['mppo_eta'],
              eta_mean=hparams['mppo_eta_mean'],
              eta_var=hparams['mppo_eta_var'],
              gamma=hparams['gamma'],
              num_action_samples=hparams['mppo_num_action_samples'],
              termination=termination)

mppo_optimizer = optim.Adam([p for name, p in mppo.named_parameters()
                             if 'model' not in name],
                            lr=hparams['model_opt_lr'],
                            weight_decay=hparams['model_opt_weight_decay'])

# %% Define Agent
comment = model.name
comment = f"{model.name} {'Optimistic' if hparams['optimistic'] else 'Expected'}"

agent = MBMPPOAgent(
    environment.name, mppo, model_optimizer, mppo_optimizer,
    initial_distribution=initial_distribution,
    action_scale=3,
    plan_horizon=hparams['plan_horizon'], plan_samples=hparams['plan_samples'],
    plan_elite=hparams['plan_elite'],
    max_memory=hparams['max_memory'], batch_size=hparams['batch_size'],
    num_model_iter=hparams['num_model_iter'],
    num_mppo_iter=hparams['num_mppo_iter'],
    num_gradient_steps=hparams['num_gradient_steps'],
    sim_num_subsample=hparams['sim_num_subsample'],
    sim_initial_states_num_trajectories=hparams['sim_initial_states_num_trajectories'],
    sim_initial_dist_num_trajectories=hparams['sim_initial_dist_num_trajectories'],
    sim_num_steps=hparams['sim_num_steps'],
    gamma=hparams['gamma'], comment=comment)

# %% Train Agent
with gpytorch.settings.fast_computations(), gpytorch.settings.fast_pred_var(), \
     gpytorch.settings.fast_pred_samples(), gpytorch.settings.memory_efficient():
    train_agent(agent, environment,
                num_episodes=hparams['train_episodes'],
                max_steps=hparams['environment_max_steps'],
                plot_flag=hparams['plot_train_results'],
                print_frequency=hparams['print_frequency'],
                render=hparams['render_train'],
                plot_callbacks=[plot_cartpole_trajectories]
                )
agent.logger.export_to_json(hparams)

# %% Test agent.
metrics = dict()
evaluate_agent(agent, environment, num_episodes=hparams['test_episodes'],
               max_steps=hparams['environment_max_steps'],
               render=hparams['render_test'])

returns = np.mean(agent.logger.get('environment_return')[-hparams['test_episodes']:])
metrics.update({"test/test_env_returns": returns})
returns = np.mean(agent.logger.get('environment_return')[:-hparams['test_episodes']])
metrics.update({"test/train_env_returns": returns})

# # Test Policy on Environment and Model.
# dynamical_model.eval()
# returns, _ = test_policy_on_environment(environment, agent.policy, test_state)
# metrics.update({"test/policy_environment": returns})
#
# returns, _ = test_policy_on_model(dynamical_model, reward_model, agent.plan_policy,
#                                   test_state)
# metrics.update({"test/policy_model": returns})
#
# # Test Mean Policy on Environment and Model.
# returns, _ = test_policy_on_environment(
#     environment,
#     lambda x: (agent.policy(x)[0][:agent.dynamical_model.base_model.dim_action],
#                torch.zeros(1)), test_state, policy_str='Expected Policy')
# metrics.update({"test/expected_policy_environment": returns})
#
# returns, _ = test_policy_on_model(
#     dynamical_model, reward_model,
#     lambda x: (agent.plan_policy(x)[0], torch.zeros(1)),
#     test_state, policy_str='Expected Policy')
# metrics.update({"test/expected_policy_model": returns})
#
# # Test Policy on Sampled Model.
# sampled_model = TransformedModel(model, transformations)
# returns, _ = test_policy_on_model(sampled_model, reward_model, agent.policy, test_state)
# metrics.update({"test/policy_sampled_model": returns})
#
# returns, _ = test_policy_on_model(
#     sampled_model, reward_model,
#     lambda x: (agent.policy(x)[0][:sampled_model.dim_action],
#                torch.zeros(1)), test_state, policy_str='Expected Policy')
# metrics.update({"test/expected_policy_sampled_model": returns})

# plot_values_and_policy(agent.value_function, agent.policy,
#                        trajectory=stack_list_of_tuples(agent.last_trajectory),
#                        num_entries=[200, 200],
#                        bounds=[(-2 * np.pi, 2 * np.pi), (-12, 12)])

agent.logger.log_hparams(hparams, metrics)
