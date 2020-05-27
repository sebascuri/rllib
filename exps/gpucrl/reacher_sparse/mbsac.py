from dotmap import DotMap

from rllib.util import RewardTransformer

from exps.gpucrl.reacher_sparse import TRAIN_EPISODES, ENVIRONMENT_MAX_STEPS, ACTION_COST, \
    get_agent_and_environment
from exps.gpucrl.mb_sac_arguments import parser
from exps.gpucrl.plotters import plot_last_rewards
from exps.gpucrl.util import train_and_evaluate

PLAN_HORIZON = 0
PLAN_SAMPLES = 500
PLAN_ELITES = 10
ALGORITHM_NUM_ITER = 50
SIM_TRAJECTORIES = 5
SIM_EXP_TRAJECTORIES = 10
SIM_MEMORY_TRAJECTORIES = 5
SIM_NUM_STEPS = 5
SIM_SUBSAMPLE = 2
SIM_TOTAL = SIM_NUM_STEPS * (
        SIM_TRAJECTORIES + SIM_EXP_TRAJECTORIES + SIM_MEMORY_TRAJECTORIES
) / SIM_SUBSAMPLE

parser.description = 'Run Reacher using Model-Based SAC.'
parser.set_defaults(
    # exploration='expected',
    action_cost=ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    plan_horizon=PLAN_HORIZON,
    plan_samples=PLAN_SAMPLES,
    plan_elites=PLAN_ELITES,

    sac_num_iter=ALGORITHM_NUM_ITER,
    sac_eta=None,
    sac_epsilon=0.2,
    sac_opt_lr=3e-4,
    sac_gradient_steps=100,
    sac_target_frequency_update=8,
    sac_num_action_samples=4,
    sac_batch_size=256,

    sim_num_steps=SIM_NUM_STEPS,
    sim_initial_states_num_trajectories=SIM_TRAJECTORIES,
    sim_initial_dist_num_trajectories=SIM_EXP_TRAJECTORIES,
    sim_memory_num_trajectories=SIM_MEMORY_TRAJECTORIES,
    sim_num_subsample=SIM_SUBSAMPLE,

    model_kind='DeterministicEnsemble',
    model_learn_num_iter=50,
    max_memory=10 * ENVIRONMENT_MAX_STEPS,
    model_layers=[256, 256, 256],
    model_non_linearity='swish',
    model_opt_lr=1e-4,
    model_opt_weight_decay=0.0005,

    policy_layers=[256, 256],
    q_function_layers=[256, 256],
)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, 'mbsac')
"The optimal reward scale varies between environments, and should be tuned for each task separately."
agent.algorithm.reward_transformer = RewardTransformer(offset=-2, scale=10 / 2,
                                                       low=0, high=10)
train_and_evaluate(agent, environment, params=params,
                   plot_callbacks=[plot_last_rewards])
