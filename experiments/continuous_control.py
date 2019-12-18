import matplotlib.pyplot as plt
from rllib.util import rollout_agent, rollout_policy
from rllib.environment.systems import InvertedPendulum, GaussianSystem
from rllib.environment import SystemEnvironment, GymEnvironment
from rllib.policy import FelixPolicy
from rllib.value_function import NNQFunction, TabularQFunction
from rllib.dataset import ExperienceReplay
from rllib.exploration_strategies import GaussianExploration
from rllib.agent import DDPGAgent
import torch.nn.functional as func
import numpy as np
import torch.optim
import pickle

ENVIRONMENT = ['MountainCarContinuous-v0', 'Pendulum-v0'][0]
NUM_EPISODES = 25
MAX_STEPS = 2500
TARGET_UPDATE_FREQUENCY = 2
TARGET_UPDATE_TAU = 0.99
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 64
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10000
LAYERS = [64, 64]
SEED = 0
RENDER = True

torch.manual_seed(SEED)
np.random.seed(SEED)

system = InvertedPendulum(mass=0.1, length=0.5, friction=0.)
# system = system.linearize()


def initial_state():
    return np.array([np.deg2rad(20), 0.])


def termination(state):
    return bool(np.abs(state[..., 0]) >= np.deg2rad(45))


def reward_function(state, action):
    theta = np.rad2deg(state[..., 0])
    reward = np.exp(- 0.5 / (10 ** 2) * theta ** 2)
    print(reward, theta)
    return reward
    # return -(np.rad2deg(state[..., 0]) ** 2)


fig, axes = plt.subplots(2, 1, sharex=False)
# environment = SystemEnvironment(system, initial_state, reward=reward_function,
#                                 termination=termination)
environment = GymEnvironment(ENVIRONMENT, SEED)

policy = FelixPolicy(environment.dim_state, environment.dim_action, temperature=0.)
exploration = GaussianExploration(EPS_START, EPS_END, EPS_DECAY)
q_function = NNQFunction(environment.dim_state, environment.dim_action,
                         num_states=environment.num_states,
                         num_actions=environment.num_actions,
                         layers=LAYERS,
                         tau=TARGET_UPDATE_TAU)
memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE, batch_size=BATCH_SIZE)
actor_optimizer = torch.optim.Adam(policy.parameters, lr=ACTOR_LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
critic_optimizer = torch.optim.Adam(q_function.parameters, lr=CRITIC_LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
criterion = func.mse_loss

agent = DDPGAgent(q_function, policy, exploration, criterion, critic_optimizer,
                  actor_optimizer, memory,
                  target_update_frequency=TARGET_UPDATE_FREQUENCY,
                  gamma=GAMMA,
                  episode_length=MAX_STEPS)
# rollout_policy(environment, agent.policy, max_steps=100, render=True)

rollout_agent(environment, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS,
              render=RENDER)

# rollout
axes[0].plot(agent.episodes_cumulative_rewards, label='DDPG')
tds = agent.logs['episode_td_errors']
axes[1].plot(tds, label='DDPG')

with open('../runs/{}_{}.pkl'.format('Pendulum', 'DDPG'), 'wb') as file:
    pickle.dump(agent, file)

axes[1].set_xlabel('Episode')
axes[0].set_ylabel('Rewards')
axes[0].legend(loc='best')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Mean Absolute TD-Error')
plt.show()

# rollout_policy(environment, agent.policy, max_steps=100, render=True)
