from dotmap import DotMap
import gpytorch
import numpy as np
import torch.jit
from torch.distributions import Uniform
from rllib.dataset.transforms import MeanFunction, ActionScaler, DeltaState, \
    AngleWrapper
from rllib.environment import GymEnvironment
from rllib.reward.mujoco_rewards import HalfCheetahReward
from rllib.util.training import train_agent, evaluate_agent
from exps.gpucrl.half_cheetah.util import StateTransform

from exps.gpucrl.util import large_state_termination, get_mpc_agent
from exps.gpucrl.plotters import plot_last_rewards
from exps.gpucrl.mpc_arguments import parser

parser.description = 'Run Half Cheetah using Model-Based MPC.'
parser.set_defaults(action_cost=0.1, environment_max_steps=1000, train_episodes=15,
                    mpc_horizon=40,
                    model_kind='DeterministicEnsemble', model_learn_num_iter=50,
                    model_opt_lr=1e-3, render_train=True)
args = parser.parse_args()
params = DotMap(vars(args))
torch.manual_seed(params.seed)
np.random.seed(params.seed)

# %% Define Helper modules
transformations = [
    # ActionScaler(scale=3),
    MeanFunction(DeltaState()),
    # AngleWrapper(indexes=[2]),
]

input_transform = StateTransform()

# %% Define Environment.
environment = GymEnvironment('MBRLHalfCheetah-v0', action_cost=params.action_cost,
                             seed=params.seed)
reward_model = HalfCheetahReward(action_cost=params.action_cost)
exploratory_distribution = torch.distributions.Uniform(
    torch.tensor([-np.pi, -1.25, -0.05, -0.05]),
    torch.tensor([+np.pi, +1.25, +0.05, +0.05])
)

agent = get_mpc_agent(environment.name, environment.dim_state, environment.dim_action,
                      params, reward_model, transformations, input_transform,
                      termination=large_state_termination,
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
                plot_callbacks=[plot_last_rewards]
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
