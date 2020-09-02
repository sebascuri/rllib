"""Solution of Cart 1-d with DDPG."""

import numpy as np
import torch.optim
from torch.distributions import Uniform

from exps.risk_averse.util import (
    Cart1dReward,
    Cart1dTermination,
    plot_cart_trajectories,
)
from rllib.agent import TD3Agent
from rllib.environment import SystemEnvironment
from rllib.environment.systems.cart1d import Cart1d
from rllib.policy import FelixPolicy
from rllib.util.parameter_decay import ExponentialDecay
from rllib.util.training import evaluate_agent, train_agent

torch.manual_seed(0)
np.random.seed(0)

# %% Define Environment.
goal_x = 2.0
dt = 0.1
reward_goal = 100.0
prob_reward_high_v = 0.2
reward_high_v = -20.0

initial_state = Uniform(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
environment = SystemEnvironment(
    Cart1d(dt),
    initial_state=lambda: np.array([0.0, 0.0]),
    reward=Cart1dReward(
        goal_x=goal_x,
        reward_goal=reward_goal,
        reward_high_v=reward_high_v,
        prob_reward_high_v=prob_reward_high_v,
    ),
    termination_model=Cart1dTermination(goal_x=goal_x),
)

# %% Define Training algorithm.
NUM_EPISODES = 100
MAX_STEPS = 200
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 500

policy = FelixPolicy.default(environment, deterministic=True)
noise = ExponentialDecay(EPS_START, EPS_END, EPS_DECAY)

agent = TD3Agent.default(environment, gamma=GAMMA, exploration_noise=noise)

train_agent(
    agent,
    environment,
    NUM_EPISODES,
    MAX_STEPS,
    plot_flag=True,
    plot_callbacks=[plot_cart_trajectories],
    plot_frequency=5,
)
evaluate_agent(agent, environment, 1, MAX_STEPS)
plot_cart_trajectories(agent, NUM_EPISODES + 1)
