import pytest
import torch.optim

from rllib.agent import DPGAgent, TD3Agent
from rllib.dataset import PrioritizedExperienceReplay
from rllib.environment import GymEnvironment
from rllib.policy import FelixPolicy
from rllib.util.parameter_decay import ExponentialDecay
from rllib.util.training import evaluate_agent, train_agent
from rllib.value_function import NNQFunction

NUM_EPISODES = 10
MAX_STEPS = 25
TARGET_UPDATE_FREQUENCY = 4
TARGET_UPDATE_TAU = 0.1
MEMORY_MAX_SIZE = 5000
BATCH_SIZE = 64

ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0

GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
POLICY_NOISE = 0.2
LAYERS = [64, 64]
SEED = 0


@pytest.fixture(params=['Pendulum-v0', 'MountainCarContinuous-v0'])
def environment(request):
    return request.param


@pytest.fixture(params=[DPGAgent, TD3Agent])
def agent(request):
    return request.param


def test_ddpg_interaction(environment, agent):
    environment = GymEnvironment(environment, SEED)

    q_function = NNQFunction(environment.dim_observation, environment.dim_action,
                             num_states=environment.num_states,
                             num_actions=environment.num_actions,
                             layers=LAYERS,
                             tau=TARGET_UPDATE_TAU,
                             )
    policy = FelixPolicy(environment.dim_state, environment.dim_action,
                         tau=TARGET_UPDATE_TAU, deterministic=True)

    criterion = torch.nn.MSELoss

    noise = ExponentialDecay(EPS_START, EPS_END, EPS_DECAY)
    memory = PrioritizedExperienceReplay(max_len=MEMORY_MAX_SIZE)
    optimizer = torch.optim.Adam([
        {'params': q_function.parameters(), 'lr': CRITIC_LEARNING_RATE},
        {'params': policy.parameters(), 'lr': ACTOR_LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)

    agent = agent(
        environment.name, q_function=q_function, policy=policy, exploration_noise=noise,
        criterion=criterion, optimizer=optimizer, memory=memory, batch_size=BATCH_SIZE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
        gamma=GAMMA, policy_noise=POLICY_NOISE)

    train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, plot_flag=False)
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)
