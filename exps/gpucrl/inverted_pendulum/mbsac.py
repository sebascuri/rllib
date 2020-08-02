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

PLAN_HORIZON, SIM_TRAJECTORIES = 0, 20

parser.description = "Run Swing-up Inverted Pendulum using Model-Based SAC."
parser.set_defaults(
    action_cost=ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    plan_horizon=PLAN_HORIZON,
    sim_num_steps=ENVIRONMENT_MAX_STEPS,
    sim_initial_states_num_trajectories=1,  # SIM_TRAJECTORIES,
    sim_initial_dist_num_trajectories=SIM_TRAJECTORIES - 1,
    model_kind="ProbabilisticEnsemble",
    model_learn_num_iter=50,
    model_opt_lr=1e-3,
    seed=0,
    sac_num_iter=30,
    sac_batch_size=1000,
    sac_gradient_steps=20,
    sim_refresh_interval=0,
    sim_max_memory=40000,
    sac_regularization=False,
    sac_opt_lr=1e-2,
    sac_eta=1.0,
    render_train=True,
)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, "mbsac")
# agent.algorithm.reward_transformer = RewardTransformer(scale=10.0)
train_and_evaluate(
    agent, environment, params, plot_callbacks=[plot_pendulum_trajectories]
)
