from rllib.algorithms.td_learning import ERLSTD
from rllib.value_function import NNValueFunction
from rllib.environment import GymEnvironment
import torch
import torch.nn.functional as func
import pickle
import matplotlib.pyplot as plt
import copy


FILE = '../environments/TaxiEnv_GQLearningAgent_199.pkl'
ENVIRONMENT = 'Taxi-v2'
BATCH_SIZE = 32
LEARNING_RATE = 0.05
MOMENTUM = 0.1
WEIGHT_DECAY = 0
GAMMA = 0.99
LAYERS = []
SEED = 0
BATCHES = 10000

with open(FILE, 'rb') as f:
    agent = pickle.load(f)

environment = GymEnvironment(ENVIRONMENT, SEED)
value_function = NNValueFunction(dim_state=environment.dim_state,
                                 num_states=environment.num_states,
                                 layers=LAYERS, biased_head=False, tau=1)

optimizer = torch.optim.SGD(value_function.parameters, lr=LEARNING_RATE,
                            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
criterion = func.mse_loss

memory = copy.deepcopy(agent._memory)
memory.batch_size = BATCH_SIZE

alg = ERLSTD(value_function, criterion, optimizer, memory, GAMMA)

td_errors, losses = alg.train(BATCHES)
plt.plot(losses)
plt.xlabel('Gradient Steps')
plt.ylabel('TD-Error')
# plt.title('{} in {}'.format(agent.name, environment.name))
plt.show()
