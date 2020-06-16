"""Working example of MPPO."""
from itertools import chain

import numpy as np
import torch
import torch.nn as nn

from rllib.agent import MPPOAgent
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.policy import FelixPolicy, NNPolicy  # noqa: F401
from rllib.util.neural_networks import zero_bias  # noqa: F401
from rllib.util.training import evaluate_agent, train_agent
from rllib.value_function import NNQFunction

# ENVIRONMENT = 'Taxi-v2'
# ENVIRONMENT = 'CartPole-v0'
ENVIRONMENT = "Pendulum-v0"

EPSILON = 0.01
EPSILON_MEAN = 0.01
EPSILON_VAR = 1e-4
NUM_EPISODES = 50
BATCH_SIZE = 100
# MAX_STEPS = 2000
MAX_STEPS = 1000

LR = 5e-4
# NUM_ITER = 20
# NUM_ROLLOUTS = 1
NUM_ITER = 500
NUM_ROLLOUTS = 2
NUM_ACTION_SAMPLES = 15

GAMMA = 0.99
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
env = GymEnvironment(ENVIRONMENT, SEED)
# policy = NNPolicy(
#     env.dim_state,
#     env.dim_action,
#     env.num_states,
#     env.num_actions,
#     squashed_output=True,
#     layers=[64, 64],
#     tau=0.02,
#     action_scale=env.action_scale,
#     biased_head=False,
# )
# zero_bias(policy)

policy = FelixPolicy(
    env.dim_state, env.dim_action, deterministic=False, action_scale=env.action_scale
)

q_function = NNQFunction(
    env.dim_state, env.dim_action, env.num_states, env.num_actions, layers=[64, 64]
)


optimizer = torch.optim.Adam(chain(policy.parameters(), q_function.parameters()), lr=LR)
memory = ExperienceReplay(max_len=int(MAX_STEPS * NUM_ROLLOUTS))
agent = MPPOAgent(
    policy=policy,
    q_function=q_function,
    criterion=nn.MSELoss,
    optimizer=optimizer,
    memory=memory,
    epsilon=EPSILON,
    epsilon_mean=EPSILON_MEAN,
    epsilon_var=EPSILON_VAR,
    num_action_samples=NUM_ACTION_SAMPLES,
    train_frequency=0,
    num_rollouts=NUM_ROLLOUTS,
    target_update_frequency=4,
    num_iter=NUM_ITER,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
)

train_agent(agent, env, NUM_EPISODES, MAX_STEPS, print_frequency=1, render=True)
evaluate_agent(agent, env, num_episodes=2, max_steps=MAX_STEPS)
