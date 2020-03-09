import matplotlib.pyplot as plt
from rllib.agent import ActorCriticAgent
from rllib.environment import GymEnvironment
from rllib.util.rollout import rollout_agent
from rllib.policy import NNPolicy
from rllib.value_function import NNQFunction
import torch
import numpy as np
import torch.nn.modules.loss as loss

ENVIRONMENT = 'CartPole-v0'
MAX_STEPS = 200
NUM_ROLLOUTS = 8
NUM_EPISODES = 500
TARGET_UPDATE_FREQUENCY = 1
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3

GAMMA = 0.99
LAYERS = [200, 200]
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment(ENVIRONMENT, SEED)
policy = NNPolicy(environment.dim_state, environment.dim_action,
                  num_states=environment.num_states,
                  num_actions=environment.num_actions,
                  layers=LAYERS)
critic = NNQFunction(environment.dim_state, environment.num_actions,
                     num_states=environment.num_states,
                     num_actions=environment.num_actions, layers=LAYERS)

actor_optimizer = torch.optim.Adam(policy.parameters(), lr=ACTOR_LEARNING_RATE)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE)
criterion = loss.MSELoss

agent = ActorCriticAgent(policy=policy, actor_optimizer=actor_optimizer, critic=critic,
                         critic_optimizer=critic_optimizer, criterion=criterion,
                         num_rollouts=NUM_ROLLOUTS, gamma=GAMMA)

rollout_agent(environment, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)

for key, log in agent.logs.items():
    plt.plot(log.episode_log)
    plt.xlabel('Episode')
    plt.ylabel(key.capitalize())
    plt.title('{} in {}'.format(agent.name, environment.name))
    plt.show()

rollout_agent(environment, agent, max_steps=MAX_STEPS, num_episodes=1, render=True)
