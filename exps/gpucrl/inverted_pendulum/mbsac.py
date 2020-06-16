"""Run the inverted-pendulum using MB-SAC."""

from dotmap import DotMap

from exps.gpucrl.inverted_pendulum import (
    ACTION_COST,
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
)
from exps.gpucrl.inverted_pendulum.plotters import plot_pendulum_trajectories
from exps.gpucrl.mb_sac_arguments import parser
from exps.gpucrl.util import train_and_evaluate
from rllib.util.utilities import RewardTransformer

PLAN_HORIZON, SIM_TRAJECTORIES = 1, 8

parser.description = "Run Swing-up Inverted Pendulum using Model-Based SAC."
parser.set_defaults(
    sac_eta=None,
    sac_epsilon=0.2,
    sac_batch_size=100,
    action_cost=ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    plan_horizon=PLAN_HORIZON,
    sim_num_steps=ENVIRONMENT_MAX_STEPS,
    sim_initial_states_num_trajectories=SIM_TRAJECTORIES // 2,
    sim_initial_dist_num_trajectories=SIM_TRAJECTORIES // 2,
    sim_refresh_interval=1,
    model_kind="ExactGP",
    model_learn_num_iter=0,
    model_opt_lr=1e-3,
)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, "mbsac")
agent.algorithm.reward_transformer = RewardTransformer(scale=10.0)
train_and_evaluate(
    agent, environment, params, plot_callbacks=[plot_pendulum_trajectories]
)
