"""Working example of Softmax vs Eps Greedy Policies."""

import matplotlib.pyplot as plt
import numpy as np
import torch.optim

from rllib.agent import DDQNAgent
from rllib.dataset import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.policy import EpsGreedy, SoftMax
from rllib.util.parameter_decay import ExponentialDecay
from rllib.util.rollout import rollout_agent
from rllib.value_function import NNQFunction

ENVIRONMENT = "CartPole-v0"
NUM_EPISODES = 50
MAX_STEPS = 200
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.1
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
LAYERS = [64, 64]
SEED = 0

for name, Policy in {"eps_greedy": EpsGreedy, "softmax": SoftMax}.items():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(ENVIRONMENT, SEED)
    q_function = NNQFunction(
        dim_state=environment.dim_observation,
        dim_action=environment.dim_action,
        num_states=environment.num_states,
        num_actions=environment.num_actions,
        layers=LAYERS,
    )
    policy = Policy(q_function, ExponentialDecay(start=1.0, end=0.01, decay=500))
    q_target = NNQFunction(
        dim_state=environment.dim_observation,
        dim_action=environment.dim_action,
        num_states=environment.num_states,
        num_actions=environment.num_actions,
        layers=LAYERS,
        tau=TARGET_UPDATE_TAU,
    )

    optimizer = torch.optim.Adam(q_function.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss
    memory = ExperienceReplay(max_len=MEMORY_MAX_SIZE)

    agent = DDQNAgent(
        critic=q_function,
        policy=policy,
        criterion=criterion,
        optimizer=optimizer,
        memory=memory,
        batch_size=BATCH_SIZE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
        gamma=GAMMA,
    )
    rollout_agent(environment, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
    print(agent)
    plt.plot(agent.logger.get("rewards"), label=name)
plt.xlabel("Episode")
plt.ylabel("Cumulative Rewards")
plt.legend(loc="best")
plt.show()
