"""Working example of SAC."""
from itertools import chain

import numpy as np
import torch.optim

from rllib.agent import SACAgent
from rllib.dataset import ExperienceReplay, PrioritizedExperienceReplay  # noqa: F401
from rllib.environment import GymEnvironment
from rllib.policy import FelixPolicy
from rllib.util.training import evaluate_agent, train_agent
from rllib.value_function import NNQFunction

ENVIRONMENT = ["MountainCarContinuous-v0", "Pendulum-v0"][1]
NUM_EPISODES = 40
MAX_STEPS = 1000
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.01
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
GAMMA = 0.99
LAYERS = [64, 64]
SEED = 1
RENDER = True

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
policy = FelixPolicy(
    environment.dim_state,
    environment.dim_action,
    deterministic=False,
    tau=TARGET_UPDATE_TAU,
    action_scale=environment.action_scale,
)

q_function = NNQFunction(
    environment.dim_state,
    environment.dim_action,
    num_states=environment.num_states,
    num_actions=environment.num_actions,
    layers=LAYERS,
    tau=TARGET_UPDATE_TAU,
)
memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE)


optimizer = torch.optim.Adam(
    chain(policy.parameters(), q_function.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
criterion = torch.nn.MSELoss

agent = SACAgent(
    q_function=q_function,
    policy=policy,
    criterion=criterion,
    optimizer=optimizer,
    memory=memory,
    batch_size=BATCH_SIZE,
    eta=1.0,
    regularization=True,
    target_update_frequency=TARGET_UPDATE_FREQUENCY,
    num_iter=1,
    train_frequency=1,
    exploration_episodes=0,
    gamma=GAMMA,
)
train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, print_frequency=1, render=True)
evaluate_agent(agent, environment, 1, MAX_STEPS)
