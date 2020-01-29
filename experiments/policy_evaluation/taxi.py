from rllib.algorithms.td_learning import TD, TDC, GTD, GTD2, TDL1, TDLinf
from rllib.value_function import NNValueFunction
from rllib.exploration_strategies import EpsGreedy
from rllib.environment import GymEnvironment
from rllib.dataset import ExperienceReplay, PrioritizedExperienceReplay, L1Sampler, \
    LinfSampler, init_er_from_er, init_er_from_environment, init_er_from_rollout

import torch
import torch.nn.functional as func
import pickle
import matplotlib.pyplot as plt
import copy


FILE = '../environments/TaxiEnv_GQLearningAgent_199.pkl'
ENVIRONMENT = 'Taxi-v2'

FILE = '../environments/CartPoleEnv_DDQNAgent_49.pkl'
ENVIRONMENT = 'CartPole-v0'

BATCH_SIZE = 128
LEARNING_RATE = 0.05
MOMENTUM = 0.1
WEIGHT_DECAY = 0
GAMMA = 0.99
LAYERS = [120]
SEED = 0
EPOCHS = 100
EPSILON = 0.1

LR_THETA = 0.001
LR_OMEGA = 1 / 4 * LR_THETA

with open(FILE, 'rb') as f:
    agent = pickle.load(f)

environment = GymEnvironment(ENVIRONMENT, SEED)
max_len = 500
sampler = ExperienceReplay(max_len, batch_size=BATCH_SIZE)

# init_er_from_environment(sampler, environment)
# init_er_from_er(sampler, agent.memory)
init_er_from_rollout(sampler, agent, environment, max_steps=200)

value_function = NNValueFunction(dim_state=environment.dim_state,
                                 num_states=environment.num_states,
                                 layers=LAYERS, biased_head=False, tau=1)

alg = GTD2(environment=environment, agent=agent, sampler=sampler,
         value_function=value_function, gamma=GAMMA, lr_theta=LR_THETA,
         lr_omega=LR_OMEGA)
td_error = alg.train(EPOCHS)
plt.plot(td_error)
plt.show()
