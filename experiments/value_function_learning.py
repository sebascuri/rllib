import matplotlib.pyplot as plt
import torch.nn.functional as func
import torch.optim
import numpy as np
from rllib.dataset import ExperienceReplay
from rllib.exploration_strategies import EpsGreedy
from rllib.value_function import NNQFunction
from rllib.environment import GymEnvironment
from rllib.agent import DDQNAgent
from rllib.util import rollout_agent, rollout_policy


ENVIRONMENT = 'CartPole-v0'
NUM_EPISODES = 50
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.9
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
LAYERS = [64, 64]
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
environment = GymEnvironment(ENVIRONMENT, SEED)

exploration = EpsGreedy(EPS_START, EPS_END, EPS_DECAY)
q_function = NNQFunction(environment.dim_observation, environment.dim_action,
                         num_states=environment.num_observation,
                         num_actions=environment.num_action,
                         layers=LAYERS
                         )
q_target = NNQFunction(environment.dim_observation, environment.dim_action,
                       num_states=environment.num_observation,
                       num_actions=environment.num_action,
                       layers=LAYERS,
                       tau=TARGET_UPDATE_TAU
                       )
optimizer = torch.optim.Adam
criterion = func.mse_loss
memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE)

hyper_params = {
    'target_update_frequency': TARGET_UPDATE_FREQUENCY,
    'batch_size': BATCH_SIZE,
    'gamma': GAMMA,
    'learning_rate': LEARNING_RATE
}
agent = DDQNAgent(q_function, q_target, exploration, criterion, optimizer, memory,
                  hyper_params)

rollout_agent(environment, agent, num_episodes=NUM_EPISODES)

policy = q_function.extract_policy(temperature=0.1)

environment = GymEnvironment(ENVIRONMENT, SEED)
trajectories = rollout_policy(environment, policy, num_episodes=NUM_EPISODES)

rewards = []
for trajectory in trajectories:
    reward = 0
    for observation in trajectory:
        reward += observation.reward
    rewards.append(reward)

plt.plot(agent.episodes_cumulative_rewards, label='Agent Learning')
plt.plot(rewards, label='Executed Policy')

plt.xlabel('Episode')
plt.ylabel('Cumulative Rewards')
plt.legend(loc='best')
plt.show()
