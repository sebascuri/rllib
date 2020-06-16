"""Run MBMPPO on Pusher."""
from dotmap import DotMap

from exps.gpucrl.mb_mppo_arguments import parser
from exps.gpucrl.plotters import plot_last_rewards
from exps.gpucrl.pusher import (
    ACTION_COST,
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
)
from exps.gpucrl.util import train_and_evaluate

PLAN_HORIZON = 4
PLAN_SAMPLES = 500
MPPO_ETA, MPPO_ETA_MEAN, MPPO_ETA_VAR = 0.5, 0.1, 0.5
MPPO_NUM_ITER = 50
SIM_TRAJECTORIES = 64
SIM_EXP_TRAJECTORIES = 0  # 32
SIM_MEMORY_TRAJECTORIES = 0  # 8
SIM_NUM_STEPS = ENVIRONMENT_MAX_STEPS

parser.description = "Run Pusher using Model-Based MPPO."
parser.set_defaults(
    # exploration='thompson',
    action_cost=ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    plan_horizon=PLAN_HORIZON,
    plan_samples=PLAN_SAMPLES,
    mppo_num_iter=MPPO_NUM_ITER,
    # mppo_eta=.5,
    # mppo_eta_mean=1.,
    # mppo_eta_var=5.,
    mppo_eta=None,
    mppo_eta_mean=None,
    mppo_eta_var=None,
    mppo_epsilon=0.1,
    mppo_epsilon_mean=0.1,
    mppo_epsilon_var=1e-4,
    sim_num_steps=SIM_NUM_STEPS,
    sim_initial_states_num_trajectories=SIM_TRAJECTORIES,
    sim_initial_dist_num_trajectories=SIM_EXP_TRAJECTORIES,
    sim_memory_num_trajectories=SIM_MEMORY_TRAJECTORIES,
    model_kind="DeterministicEnsemble",
    model_learn_num_iter=5,
    max_memory=ENVIRONMENT_MAX_STEPS,
    model_layers=[200, 200, 200],
    model_non_linearity="swish",
    model_opt_lr=1e-4,
    model_opt_weight_decay=0.0005,
    mppo_opt_lr=5e-4,
    mppo_gradient_steps=50,
    policy_layers=[100, 100],
    value_function_layers=[200, 200],
)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, "mbmppo")
agent.mppo_target_update_frequency = 20
agent.dynamical_model.beta = 5.0
# agent.exploration_episodes = 10
train_and_evaluate(
    agent, environment, params=params, plot_callbacks=[plot_last_rewards]
)
