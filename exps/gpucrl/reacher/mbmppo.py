from dotmap import DotMap

from exps.gpucrl.reacher import TRAIN_EPISODES, ENVIRONMENT_MAX_STEPS, ACTION_COST, \
    get_agent_and_environment
from exps.gpucrl.mb_mppo_arguments import parser
from exps.gpucrl.plotters import plot_last_rewards
from exps.gpucrl.util import train_and_evaluate
from rllib.util.utilities import RewardTransformer

PLAN_HORIZON = 20
PLAN_SAMPLES = 500
PLAN_ELITES = 10
ALGORITHM_NUM_ITER = 50
SIM_TRAJECTORIES = 25
SIM_EXP_TRAJECTORIES = 25
SIM_MEMORY_TRAJECTORIES = 25
SIM_NUM_STEPS = 10

parser.description = 'Run Reacher using Model-Based MPPO.'
parser.set_defaults(
    # exploration='expected',
    action_cost=ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    plan_horizon=PLAN_HORIZON,
    plan_samples=PLAN_SAMPLES,
    plan_elites=PLAN_ELITES,

    mppo_num_iter=ALGORITHM_NUM_ITER,
    mppo_eta=1.,
    mppo_eta_mean=1.,
    mppo_eta_var=5.,
    # mppo_eta=None,
    # mppo_eta_mean=None,
    # mppo_eta_var=None,
    # mppo_epsilon=0.1 ,
    # mppo_epsilon_mean=0.1,
    # mppo_epsilon_var=0.0001,
    mppo_opt_lr=5e-4,
    mppo_batch_size=100,
    mppo_gradient_steps=200,
    mppo_target_frequency_update=10,

    sim_num_steps=SIM_NUM_STEPS,
    sim_initial_states_num_trajectories=SIM_TRAJECTORIES,
    sim_initial_dist_num_trajectories=SIM_EXP_TRAJECTORIES,
    sim_memory_num_trajectories=SIM_MEMORY_TRAJECTORIES,

    model_kind='DeterministicEnsemble',
    model_learn_num_iter=50,
    max_memory=10 * ENVIRONMENT_MAX_STEPS,
    model_layers=[200, 200, 200, 200],
    model_non_linearity='Tanh',
    model_opt_lr=1e-4,
    model_opt_weight_decay=0.0005,

    policy_layers=[200, 200],
    value_function_layers=[200, 200],
    value_function_tau=0.01,
)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, 'mbmppo')
agent.algorithm.reward_transformer = RewardTransformer(offset=-15, scale=1 / 15,
                                                       low=0, high=1)
# agent.exploration_episodes = 3
train_and_evaluate(agent, environment, params=params,
                   plot_callbacks=[plot_last_rewards])
