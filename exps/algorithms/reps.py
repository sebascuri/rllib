"""Working example of REPS."""
import numpy as np
import torch

from rllib.agent import REPSAgent
from rllib.algorithms.reps import QREPS, REPS  # noqa: F401
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.policy import NNPolicy
from rllib.util.training import evaluate_agent, train_agent
from rllib.value_function import NNQFunction, NNValueFunction

# ETA = 0.5
ETA = 2.0
NUM_EPISODES = 100
BATCH_SIZE = 100
LR = 1e-4
NUM_ITER = 200
NUM_ACTION_SAMPLES = 15
NUM_ROLLOUTS = 15

GAMMA = 1
SEED = 0
ENVIRONMENT = "CartPole-v0"
MAX_STEPS = 200

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
policy = NNPolicy(
    environment.dim_state,
    environment.dim_action,
    environment.num_states,
    environment.num_actions,
    layers=[64, 64],
)
value_function = NNValueFunction(
    environment.dim_state, environment.num_states, layers=[64, 64]
)
q_function = NNQFunction(
    environment.dim_state,
    environment.dim_action,
    environment.num_states,
    environment.num_actions,
    layers=[64, 64],
)

reps_loss = REPS(policy, value_function, eta=ETA, gamma=GAMMA)
# reps_loss = QREPS(policy, value_function, q_function,
#                   num_action_samples=NUM_ACTION_SAMPLES, eta=ETA, gamma=GAMMA)
optimizer = torch.optim.Adam(reps_loss.parameters(), lr=LR, weight_decay=0)

memory = ExperienceReplay(max_len=int(NUM_ROLLOUTS * MAX_STEPS))
agent = REPSAgent(
    reps_loss,
    optimizer,
    memory,
    num_iter=NUM_ITER,
    num_rollouts=NUM_ROLLOUTS,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
)

train_agent(agent, environment, NUM_EPISODES, MAX_STEPS + 1)
evaluate_agent(agent, environment, 1, MAX_STEPS + 1)
