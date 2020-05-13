from dotmap import DotMap

from exps.gpucrl.cart_pole.util import get_agent_and_environment
from exps.gpucrl.mb_mppo_arguments import parser
from exps.gpucrl.plotters import plot_last_sim_and_real_trajectory
from exps.gpucrl.util import train_and_evaluate

parser.description = 'Run Swing-up Cart-Pole using Model-Based MPPO.'
parser.set_defaults(action_cost=0.05,
                    environment_max_steps=200, train_episodes=20,
                    # exploration='expected',  # default optimistic
                    # exploration='thompson',
                    # mppo_num_iter=0,
                    plan_horizon=4,  # default 1
                    mppo_gradient_steps=100,  # default 50
                    mppo_eta=1.,  # default 1.0
                    mppo_eta_mean=1.7,  # default 1.7
                    mppo_eta_var=1.1,  # default 1.1
                    sim_num_steps=200,  # default 400
                    sim_initial_states_num_trajectories=4,  # default 4
                    sim_initial_dist_num_trajectories=8,  # default 4
                    model_kind='DeterministicEnsemble', model_learn_num_iter=50,
                    model_opt_lr=1e-3, render_train=True)

args = parser.parse_args()
params = DotMap(vars(args))
environment, agent = get_agent_and_environment(params, 'mbmppo')

train_and_evaluate(agent, environment, params,
                   plot_callbacks=[plot_last_sim_and_real_trajectory])
