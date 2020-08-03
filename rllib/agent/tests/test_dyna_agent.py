import numpy as np
import pytest
import torch

from rllib.agent import DynaAgent
from rllib.environment import GymEnvironment
from rllib.util.training import evaluate_agent, train_agent

MAX_STEPS = 25
NUM_EPISODES = 25
TARGET_UPDATE_FREQUENCY = 1
NUM_ROLLOUTS = 1
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-2

GAMMA = 0.99
LAYERS = [200, 200]
SEED = 0


@pytest.fixture(params=["MountainCarContinuous-v0"])
def continuous_environment(request):
    return request.param


@pytest.fixture(params=["DPG", "TD3", "SAC", "REPS", "MPO", "Random", "VMPO"])
def base_agent(request):
    return request.param


def rollout_agent(environment, base_agent):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(environment, SEED)
    agent = DynaAgent.default(environment, base_agent_name=base_agent, test=True)
    train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, plot_flag=False)
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)


def test_continuous_agent(continuous_environment, base_agent):
    rollout_agent(continuous_environment, base_agent)
