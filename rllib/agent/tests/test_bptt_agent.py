import numpy as np
import pytest
import torch

from rllib.agent import BPTTAgent, SVGAgent
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent

MAX_STEPS = 25
NUM_EPISODES = 4
SEED = 0


@pytest.fixture(params=["MountainCarContinuous-v0"])
def continuous_environment(request):
    return request.param


@pytest.fixture(params=[BPTTAgent, SVGAgent])
def agent(request):
    return request.param


def rollout_agent(environment, agent):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(environment, SEED)
    agent = agent.default(environment, test=True)
    train_agent(agent, environment, NUM_EPISODES, MAX_STEPS, plot_flag=False)
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)


def test_continuous_agent(continuous_environment, agent):
    rollout_agent(continuous_environment, agent)
