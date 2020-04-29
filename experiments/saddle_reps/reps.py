"""Vanilla REPS/Q-REPS experiments."""
from itertools import chain

import numpy as np
import torch
from torch.optim.lbfgs import LBFGS
from torch.optim.adam import Adam

from rllib.agent.reps_agent import REPSAgent
from rllib.algorithms.reps import REPS, QREPS
from rllib.environment import GymEnvironment
from rllib.policy import NNPolicy, SoftMax

from rllib.util.parameter_decay import Constant

from rllib.value_function import NNValueFunction, NNQFunction
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.util.training import train_agent, evaluate_agent

ETA = 1.
ALGORITHM = 'REPS'
NUM_EPISODES = 100
BATCH_SIZE = 100
LR = 1e-4
NUM_ROLLOUTS = 15
MEMORY_SIZE = 3500

GAMMA = 1
SEED = 0
ENVIRONMENT = 'CartPole-v0'
MAX_STEPS = 200

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)

value_function = NNValueFunction(environment.dim_state, environment.num_states,
                                 layers=[64, 64])
q_function = NNQFunction(environment.dim_state, environment.dim_action,
                         environment.num_states, environment.num_actions,
                         layers=[64, 64])

if ALGORITHM.upper() == 'REPS':
    policy = NNPolicy(environment.dim_state, environment.dim_action,
                      environment.num_states, environment.num_actions,
                      layers=[64, 64])
    reps_loss = REPS(policy, value_function, eta=ETA, gamma=GAMMA)
    NUM_POLICY_ITER = 2000
    NUM_DUAL_ITER = 2000
elif ALGORITHM.upper() == 'Q-REPS':
    NUM_POLICY_ITER = 0
    NUM_DUAL_ITER = 2000
    reps_loss = QREPS(value_function, q_function, eta=ETA, gamma=GAMMA)
else:
    raise NotImplementedError(f"{ALGORITHM} not implemented.")

optimizer = Adam(reps_loss.parameters(), lr=LR)

memory = ExperienceReplay(max_len=MEMORY_SIZE)
agent = REPSAgent(environment.name, reps_loss, optimizer, memory,
                  num_dual_iter=NUM_DUAL_ITER, num_policy_iter=NUM_POLICY_ITER,
                  num_rollouts=NUM_ROLLOUTS, batch_size=BATCH_SIZE,
                  gamma=GAMMA, comment=ALGORITHM)

train_agent(agent, environment, NUM_EPISODES, MAX_STEPS + 1, plot_flag=False)
evaluate_agent(agent, environment, 1, MAX_STEPS + 1)
