from dotmap import DotMap

from exps.gpucrl.cart_pole import TRAIN_EPISODES, ENVIRONMENT_MAX_STEPS, ACTION_COST, \
    get_agent_and_environment
from exps.gpucrl.mb_mppo_arguments import parser
from exps.gpucrl.plotters import plot_last_sim_and_real_trajectory
from exps.gpucrl.util import train_and_evaluate

PLAN_HORIZON = 4
MPPO_ETA, MPPO_ETA_MEAN, MPPO_ETA_VAR = 0.5, .7, 5.0
MPPO_NUM_ITER = 50  #200
SIM_TRAJECTORIES = 8
SIM_EXP_TRAJECTORIES = 32
SIM_NUM_STEPS = ENVIRONMENT_MAX_STEPS

parser.description = 'Run Cart-Pole using Model-Based MPPO.'
parser.set_defaults(action_cost=ACTION_COST,
                    train_episodes=TRAIN_EPISODES,
                    environment_max_steps=ENVIRONMENT_MAX_STEPS,
                    plan_horizon=PLAN_HORIZON,
                    mppo_num_iter=MPPO_NUM_ITER,
                    mppo_eta=MPPO_ETA,
                    mppo_eta_mean=MPPO_ETA_MEAN,
                    mppo_eta_var=MPPO_ETA_VAR,
                    mppo_gradient_steps=200,
                    sim_num_steps=SIM_NUM_STEPS,
                    sim_initial_states_num_trajectories=SIM_TRAJECTORIES,
                    sim_initial_dist_num_trajectories=SIM_EXP_TRAJECTORIES,
                    model_kind='DeterministicEnsemble',
                    model_learn_num_iter=50,
                    model_opt_lr=1e-3)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, 'mbmppo')
train_and_evaluate(agent, environment, params,
                   plot_callbacks=[plot_last_sim_and_real_trajectory])
