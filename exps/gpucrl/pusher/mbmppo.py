from dotmap import DotMap

from exps.gpucrl.pusher import TRAIN_EPISODES, ENVIRONMENT_MAX_STEPS, ACTION_COST, \
    get_agent_and_environment
from exps.gpucrl.mb_mppo_arguments import parser
from exps.gpucrl.plotters import plot_last_action_rewards
from exps.gpucrl.util import train_and_evaluate

PLAN_HORIZON, SIM_TRAJECTORIES = 4, 64

parser.description = 'Run Pusher using Model-Based MPPO.'
parser.set_defaults(action_cost=ACTION_COST,
                    train_episodes=TRAIN_EPISODES,
                    environment_max_steps=ENVIRONMENT_MAX_STEPS,
                    plan_horizon=PLAN_HORIZON,
                    sim_num_steps=ENVIRONMENT_MAX_STEPS,
                    sim_initial_states_num_trajectories=SIM_TRAJECTORIES // 2,
                    sim_initial_dist_num_trajectories=SIM_TRAJECTORIES // 2,
                    model_kind='DeterministicEnsemble',
                    model_learn_num_iter=50,
                    model_opt_lr=1e-3)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, 'mbmppo')
train_and_evaluate(agent, environment, params=params,
                   plot_callbacks=[plot_last_action_rewards])
