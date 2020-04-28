"""Solution of Cart 1-d with DDPG."""

import torch.nn.functional as func
import torch.optim
import numpy as np

from torch.distributions import Uniform
from rllib.environment import SystemEnvironment
from rllib.environment.systems.cart1d import Cart1d

from rllib.util.training import train_agent, evaluate_agent
from rllib.agent import TD3Agent
from rllib.dataset import ExperienceReplay
from rllib.exploration_strategies import GaussianNoise
from rllib.policy import FelixPolicy
from rllib.value_function import NNQFunction

from experiments.risk_averse.util import Cart1dReward, Cart1dTermination, \
    plot_cart_trajectories

torch.manual_seed(0)
np.random.seed(0)

# %% Define Environment.
goal_x = 2.0
dt = .1
reward_goal = 100.
prob_reward_high_v = .2
reward_high_v = -20.

initial_state = Uniform(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
environment = SystemEnvironment(
    Cart1d(dt), initial_state=lambda: np.array([0.0, 0.0]),
    reward=Cart1dReward(goal_x=goal_x,
                        reward_goal=reward_goal,
                        reward_high_v=reward_high_v,
                        prob_reward_high_v=prob_reward_high_v),
    termination=Cart1dTermination(goal_x=goal_x)
)

# %% Define Training algorithm.
NUM_EPISODES = 100
MAX_STEPS = 200
TARGET_UPDATE_FREQUENCY = 2
TARGET_UPDATE_TAU = 0.99
MEMORY_MAX_SIZE = 25000
BATCH_SIZE = 64
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 500
LAYERS = [64, 64]
SEED = 0
RENDER = True

policy = FelixPolicy(environment.dim_state, environment.dim_action, deterministic=True,
                     tau=TARGET_UPDATE_TAU)
noise = GaussianNoise(EPS_END)

q_function = NNQFunction(environment.dim_state, environment.dim_action,
                         num_states=environment.num_states,
                         num_actions=environment.num_actions,
                         layers=LAYERS,
                         tau=TARGET_UPDATE_TAU)
memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE)

# policy = torch.jit.script(policy)
# q_function = torch.jit.script(q_function)

actor_optimizer = torch.optim.Adam(policy.parameters(), lr=ACTOR_LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
critic_optimizer = torch.optim.Adam(q_function.parameters(), lr=CRITIC_LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss

agent = TD3Agent(
    environment.name, q_function, policy, noise, criterion, critic_optimizer,
    actor_optimizer, memory, batch_size=BATCH_SIZE,
    target_update_frequency=TARGET_UPDATE_FREQUENCY, exploration_episodes=1,
    gamma=GAMMA)

train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, plot_flag=True,
            plot_callbacks=[plot_cart_trajectories], plot_frequency=5)
evaluate_agent(agent, environment, 1, MAX_STEPS)
plot_cart_trajectories(agent, NUM_EPISODES+1)
