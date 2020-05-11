import gpytorch
import numpy as np
import torch.jit
from torch.distributions import Uniform
from dotmap import DotMap

from rllib.dataset.transforms import MeanFunction, ActionClipper, DeltaState
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.util.training import train_agent, evaluate_agent

from experiments.gpucrl.inverted_pendulum.plotters import plot_pendulum_trajectories, \
    plot_values_and_policy
from experiments.gpucrl.inverted_pendulum.util import StateTransform, PendulumReward
from experiments.gpucrl.mpc_arguments import parser
from experiments.gpucrl.util import get_mpc_agent, large_state_termination

# Parse command line Arguments.
args = parser.parse_args()
params = DotMap(vars(args))
torch.manual_seed(params.seed)
np.random.seed(params.seed)

# %% Define Helper modules
transformations = [ActionClipper(-1, 1), MeanFunction(DeltaState())]
input_transform = StateTransform()

# %% Define Environmen and Agent.
reward_model = PendulumReward(action_cost=params.action_cost)
initial_distribution = torch.distributions.Uniform(
    torch.tensor([np.pi, -0.0]), torch.tensor([np.pi, +0.0]))
exploratory_distribution = torch.distributions.Uniform(
        torch.tensor([-np.pi, -0.005]), torch.tensor([np.pi, +0.005])),

environment = SystemEnvironment(
    InvertedPendulum(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80),
    reward=reward_model,
    initial_state=initial_distribution.sample,
    termination=large_state_termination)

agent = get_mpc_agent(environment.name, environment.dim_state, environment.dim_action,
                      params, reward_model, input_transform=input_transform,
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
                plot_callbacks=[plot_pendulum_trajectories]
                )
agent.logger.export_to_json(params.toDict())

# %% Test agent.
metrics = dict()
test_state = torch.tensor(np.array([np.pi, 0.]), dtype=torch.get_default_dtype())

environment.state = test_state.numpy()
environment.initial_state = lambda: test_state.numpy()
evaluate_agent(agent, environment,
               num_episodes=params.test_episodes,
               max_steps=params.environment_max_steps,
               render=params.render_test)

returns = np.mean(agent.logger.get('environment_return')[-params.test_episodes:])
metrics.update({"test/test_env_returns": returns})
returns = np.mean(agent.logger.get('environment_return')[:-params.test_episodes])
metrics.update({"test/train_env_returns": returns})

plot_values_and_policy(agent.value_function, agent.policy,
                       trajectory=stack_list_of_tuples(agent.last_trajectory),
                       num_entries=[200, 200],
                       bounds=[(-2 * np.pi, 2 * np.pi), (-12, 12)])

agent.logger.log_hparams(params.toDict(), metrics)
