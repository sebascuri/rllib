from dotmap import DotMap

from exps.gpucrl.inverted_pendulum import TRAIN_EPISODES, ENVIRONMENT_MAX_STEPS, \
    ACTION_COST, \
    get_agent_and_environment
from exps.gpucrl.mb_sac_arguments import parser
from exps.gpucrl.inverted_pendulum.plotters import plot_pendulum_trajectories
from exps.gpucrl.util import train_and_evaluate

PLAN_HORIZON, SIM_TRAJECTORIES = 1, 8

parser.description = 'Run Swing-up Inverted Pendulum using Model-Based SAC.'
parser.set_defaults(
    sac_alpha=0.2,
    sac_batch_size=100,
    action_cost=ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    plan_horizon=PLAN_HORIZON,
    sim_num_steps=ENVIRONMENT_MAX_STEPS,
    sim_initial_states_num_trajectories=SIM_TRAJECTORIES // 2,
    sim_initial_dist_num_trajectories=SIM_TRAJECTORIES // 2,
    model_kind='ExactGP',
    model_learn_num_iter=0,
    model_opt_lr=1e-3)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, 'mbsac')
train_and_evaluate(agent, environment, params,
                   plot_callbacks=[plot_pendulum_trajectories])
