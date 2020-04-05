"""Python Script Template."""
from rllib.algorithms.reps import REPS
from rllib.environment import GymEnvironment
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction
from rllib.dataset.experience_replay import ExperienceReplay

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.rollout import step
from rllib.util.neural_networks.utilities import disable_gradient
from rllib.util.utilities import tensor_to_distribution
from rllib.util.parameter_decay import Constant, PolynomialDecay
import numpy as np
import torch

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

exp_idx = 1
EPSILON = [0.1, None, None][exp_idx]
ETA = [None, Constant(0.5), PolynomialDecay(1., 10, 2.)][exp_idx]
NUM_EPISODES = 100
BATCH_SIZE = 100
MAX_ITER = 200
LR = 1e-4
NUM_ITER = int(200)
NUM_ROLLOUTS = 15

GAMMA = 1
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
env = GymEnvironment('CartPole-v0', SEED)
policy = NNPolicy(env.dim_state, env.dim_action, env.num_states, env.num_actions,
                  layers=[64, 64])
value_function = NNValueFunction(env.dim_state, env.num_states, layers=[64, 64])

reps_loss = REPS(policy, value_function, epsilon=EPSILON, eta=ETA, gamma=GAMMA)
optimizer = torch.optim.Adam(reps_loss.parameters(), lr=LR,  weight_decay=0)

memory = ExperienceReplay(max_len=int(NUM_ROLLOUTS * MAX_ITER))
total_rewards = []
losses_ = []

for i_episode in range(NUM_EPISODES):
    state = env.reset()
    done = False

    rewards = 0
    i = 0
    while not done:
        i += 1
        pi = tensor_to_distribution(
            policy(torch.tensor(state, dtype=torch.get_default_dtype())))
        action = pi.sample().numpy()
        observation, state, done = step(env, state, action, pi, render=False)
        memory.append(observation)
        rewards += observation.reward.item()
        if i > MAX_ITER:
            break

    total_rewards.append(rewards)
    print(rewards)

    if i_episode % NUM_ROLLOUTS == NUM_ROLLOUTS - 1:
        loader = DataLoader(memory, batch_size=BATCH_SIZE, shuffle=True)
        for i in range(NUM_ITER):
            for obs, idx, weight in loader:
                loss = reps_loss(obs.state, obs.action, obs.reward, obs.next_state,
                                 obs.done)
                optimizer.zero_grad()
                loss.dual.backward()
                optimizer.step()

        print(i_episode, loss.eta.item())

        for i in range(NUM_ITER):
            for obs, idx, weight in loader:
                loss = reps_loss(obs.state, obs.action, obs.reward, obs.next_state,
                                 obs.done)

                optimizer.zero_grad()
                loss.policy_nll.backward()
                optimizer.step()

        memory.reset()

        reps_loss.update_eta()

plt.plot(total_rewards)
plt.show()
