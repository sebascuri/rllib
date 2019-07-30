import matplotlib.pyplot as plt
from rllib.agent import REINFORCE
from rllib.environment import GymEnvironment
from rllib.util.rollout import rollout_agent
from rllib.policy import NNPolicy
import torch
import numpy as np

ENVIRONMENT = 'CartPole-v0'
NUM_EPISODES = 2000
LEARNING_RATE = 0.001
GAMMA = 0.99
LAYERS = [64]
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
policy = NNPolicy(environment.dim_state, environment.dim_action,
                  num_states=environment.num_observation,
                  num_actions=environment.num_action,
                  layers=LAYERS)

optimizer = torch.optim.Adam
hyper_params = {
        'gamma': GAMMA,
        'learning_rate': LEARNING_RATE
    }
agent = REINFORCE(policy, optimizer, hyper_params)
rollout_agent(environment, agent, num_episodes=NUM_EPISODES)

plt.plot(agent.episodes_steps, label=str(agent))
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.legend(loc='best')
plt.show()
