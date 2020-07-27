"""Working example of ACTOR-CRITIC."""

import numpy as np
import torch
import torch.nn.modules.loss as loss

from rllib.agent import GAACAgent
from rllib.environment import GymEnvironment
from rllib.policy import NNPolicy
from rllib.util.training import evaluate_agent, train_agent
from rllib.value_function import NNValueFunction

ENVIRONMENT = "CartPole-v0"
MAX_STEPS = 200
NUM_ROLLOUTS = 4
NUM_EPISODES = 1000
TARGET_UPDATE_FREQUENCY = 1
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3

GAMMA = 0.99
LAYERS = [200, 200]
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
policy = NNPolicy(
    environment.dim_state,
    environment.dim_action,
    num_states=environment.num_states,
    num_actions=environment.num_actions,
    layers=LAYERS,
)
critic = NNValueFunction(
    environment.dim_state, num_states=environment.num_states, layers=LAYERS
)

optimizer = torch.optim.Adam(
    [
        {"params": policy.parameters(), "lr": ACTOR_LEARNING_RATE},
        {"params": critic.parameters(), "lr": CRITIC_LEARNING_RATE},
    ]
)
criterion = loss.MSELoss

agent = GAACAgent(
    policy=policy,
    optimizer=optimizer,
    critic=critic,
    criterion=criterion,
    num_rollouts=NUM_ROLLOUTS,
    gamma=GAMMA,
    lambda_=0.97,
)

train_agent(agent, environment, NUM_EPISODES, MAX_STEPS)
evaluate_agent(agent, environment, 1, MAX_STEPS)
