"""Python Script Template."""
from rllib.agent.mppo_agent import MPPOAgent
from rllib.algorithms.mppo import MPPO
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.policy import NNPolicy
from rllib.value_function import NNQFunction
from rllib.util.training import train_agent, evaluate_agent

import numpy as np
import torch

# ENVIRONMENT = 'Taxi-v2'
ENVIRONMENT = 'CartPole-v0'

EPSILON = 0.01
EPSILON_MEAN = 0.01
NUM_EPISODES = 250
BATCH_SIZE = 100
# MAX_STEPS = 2000
MAX_STEPS = 200

LR = 1e-4
# NUM_ITER = 20
# NUM_ROLLOUTS = 1
NUM_ITER = 200
NUM_ROLLOUTS = 5
NUM_ACTION_SAMPLES = 15

GAMMA = 0.99
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
env = GymEnvironment(ENVIRONMENT, SEED)
policy = NNPolicy(env.dim_state, env.dim_action, env.num_states, env.num_actions,
                  layers=[64, 64])
q_function = NNQFunction(env.dim_state, env.dim_action, env.num_states,
                         env.num_actions, layers=[64, 64])

mppo = MPPO(policy, q_function, num_action_samples=NUM_ACTION_SAMPLES,
            epsilon=EPSILON, epsilon_mean=EPSILON_MEAN, epsilon_var=0.,
            gamma=GAMMA)
optimizer = torch.optim.Adam(mppo.parameters(), lr=LR)
memory = ExperienceReplay(max_len=int(MAX_STEPS * NUM_ROLLOUTS), batch_size=BATCH_SIZE)
agent = MPPOAgent(ENVIRONMENT, mppo, optimizer, memory, num_rollouts=NUM_ROLLOUTS,
                  num_iter=NUM_ITER, gamma=GAMMA)

train_agent(agent, env, NUM_EPISODES, MAX_STEPS)
evaluate_agent(agent, env, num_episodes=2, max_steps=MAX_STEPS, render=True)
