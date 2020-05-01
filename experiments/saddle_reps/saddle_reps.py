"""Saddle Point Q-REPS experiments."""

import numpy as np
import torch
from torch.optim.adam import Adam

from rllib.agent.reps_agent import REPSAgent
from rllib.algorithms.reps import REPS, QREPS
from rllib.environment import GymEnvironment, SystemEnvironment
from rllib.environment.vectorized import VectorizedCartPoleEnv

from rllib.policy import NNPolicy

from rllib.value_function import NNValueFunction, NNQFunction
from rllib.dataset.experience_replay import EXP3ExperienceReplay
from rllib.util.utilities import tensor_to_distribution
from rllib.util.neural_networks.utilities import deep_copy_module
from rllib.util.parameter_decay import Constant
from rllib.util.rollout import step
from rllib.util.training import train_agent, evaluate_agent
import matplotlib.pyplot as plt

LAYERS = [64, 64]
ETA = 2.
NUM_EPISODES = 100
BATCH_SIZE = 64
MAX_STEPS = 201
NUM_ITER = 1000
NUM_ROLLOUTS = 15

GAMMA = 1
SEED = 0

SIMULATOR = True
FEATURES_XA = False

LR_SGD = 1e-4
MIXING_EXP3 = 0.1
LR_EXP3 = 1e-2


torch.manual_seed(SEED)
np.random.seed(SEED)

environment = GymEnvironment('VDiscrete-CartPole-v0', SEED)

value_function = NNValueFunction(environment.dim_state, environment.num_states,
                                 layers=LAYERS)
q_function = NNQFunction(environment.dim_state, environment.dim_action,
                         environment.num_states, environment.num_actions,
                         layers=LAYERS)

memory = EXP3ExperienceReplay(max_len=NUM_ROLLOUTS * MAX_STEPS,
                              alpha=LR_EXP3, beta=MIXING_EXP3)

reps = QREPS(value_function, q_function, eta=ETA, gamma=GAMMA)
optimizer = torch.optim.Adam(reps.parameters(), lr=LR_SGD, weight_decay=0)

total_rewards = []
losses_ = []
td_ = []
adv_ = []
mean_td_ = []
mean_adv_ = []
saddle = []
dual = []


def run_episode(environment, reps, memory, total_rewards):
    state = environment.reset()
    done = False

    rewards = 0
    i = 0
    while not done:
        i += 1
        pi = tensor_to_distribution(
            reps.policy(torch.tensor(state, dtype=torch.get_default_dtype())))
        action = pi.sample().numpy()
        observation, state, done = step(environment, state, action, pi, render=False)
        memory.append(observation)
        rewards += observation.reward.item()
        if i > MAX_STEPS:
            break
    total_rewards.append(rewards)
    print(rewards)


def get_sarsd(obs, policy):
    state = obs.state.squeeze()
    if SIMULATOR and not FEATURES_XA:
        pi = tensor_to_distribution(policy(state))
        action = pi.sample()
        y_0 = pi.probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    else:
        action = obs.action.squeeze()
        y_0 = torch.ones(BATCH_SIZE)

    if SIMULATOR:
        environment.state = state
        next_state, reward, done, _ = environment.step(action)
        done = done.float()
    else:
        next_state = obs.next_state.squeeze()
        reward = obs.reward.squeeze()
        done = obs.done.squeeze()

    return state, action, reward, next_state, done, y_0


for i_episode in range(NUM_EPISODES):
    run_episode(environment, reps, memory, total_rewards)
    if i_episode % NUM_ROLLOUTS == NUM_ROLLOUTS - 1:
        # The td-sampler samples X_k, A_k, R_k, X'_k used to compute the td-error
        # defined as td = R_k + \gamma V(X'_k) - Q(X_k, A_k) and maintains weights z_k.
        sampler_td = EXP3ExperienceReplay.from_other(memory)

        # The adv-sampler samples X_k, A_k, R_k, X'_k used to compute the advantage
        # defined as adv = Q(X_k, A_k) - V(X_k) and maintains weights y_k.
        sampler_adv = EXP3ExperienceReplay.from_other(memory)

        # Initialize priorities to zero.
        k = len(sampler_td)

        sampler_td.priorities = torch.zeros(k)
        sampler_adv.priorities = torch.zeros(k)
        # Set Learning rates:
        # exp3_lr = min(1 / k, np.sqrt(np.log(k) / (k * NUM_ITER)))
        # print(exp3_lr)
        # sampler_td.alpha = Constant(exp3_lr)
        # sampler_adv.alpha = Constant(exp3_lr)

        policy = deep_copy_module(reps.policy)
        # policy = reps.policy

        for i in range(NUM_ITER):
            # Sample from td and adv samplers.
            obs_td, idx_td, weight_td = sampler_td.get_batch(BATCH_SIZE)
            obs_adv, idx_adv, weight_adv = sampler_adv.get_batch(BATCH_SIZE)

            optimizer.zero_grad()
            # Calculate TD.
            state, action, reward, next_state, done, y_td_0 = get_sarsd(obs_td, policy)
            td = reps(state, action, reward, next_state, done).td
            y_td = sampler_td.probabilities[idx_td]

            # Calculate Advantage.
            state, action, reward, next_state, done, y_adv_0 = get_sarsd(obs_td, policy)
            adv = reps(state, action, reward, next_state, done).advantage
            y_adv = sampler_adv.probabilities[idx_adv]

            # Compute Q and V gradients.
            (td + adv).mean().backward()

            # Update Q and V
            optimizer.step()

            # Update Sampler.
            sampler_td.update(
                idx_td,
                (td.squeeze() - (torch.log(y_td / y_td_0) + 1) / reps.eta()).detach()
            )

            sampler_adv.update(
                idx_adv,
                (adv.squeeze() - (torch.log(y_adv / y_adv_0) + 1) / reps.eta()).detach()
            )

            # Log Data
            with torch.no_grad():
                td_.append(td.mean().item())
                adv_.append(adv.mean().item())

                s = y_td * td.squeeze() + y_adv * adv.squeeze()
                s -= 1. / reps.eta() * (y_td * torch.log(y_td / y_td_0)
                                        + y_adv * torch.log(y_adv / y_adv_0))
                saddle.append(s.mean())

                obs, idx, weight = memory.get_batch(BATCH_SIZE)
                loss = reps(obs.state, obs.action, obs.reward, obs.next_state, obs.done)
                mean_td_.append(loss.td.mean().item())
                mean_adv_.append(loss.advantage.mean().item())
                dual.append(loss.dual.mean().item())

        memory.reset()
        reps.update_eta()
        print(torch.sort(sampler_adv.history, descending=True)[0][:10])
        print(torch.sort(sampler_td.history, descending=True)[0][:10])

plt.plot(total_rewards)
plt.title("Total Rewards")
plt.xlabel("Num Episode")
plt.show()

fig, ax = plt.subplots(2, 1, sharex='row')
ax[0].plot(td_, 'b', label="TD = r + V(x') - Q(x, a)")
ax[0].plot(adv_, 'r', label="Advantage = Q(x, a) - V(x)")
ax[0].set_ylabel("Adversarial Samples")

ax[1].plot(mean_td_, 'b', label="TD = r + V(x') - Q(x, a)")
ax[1].plot(mean_adv_, 'r', label="Advantage = Q(x, a) - V(x)")
ax[1].set_ylabel("Average Samples")
ax[1].set_xlabel("Num Iteration")
ax[0].legend()
plt.show()

fig, ax = plt.subplots(2, 1, sharex='row')
ax[0].plot(saddle)
ax[0].set_ylabel("Saddle Objective")

ax[1].plot(dual)
ax[1].set_ylabel("Dual Objective")
ax[1].set_xlabel("Num Iteration")
plt.show()
