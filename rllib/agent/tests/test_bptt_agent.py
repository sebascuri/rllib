import numpy as np
import pytest
import torch

from rllib.agent import BPTTAgent, SVGAgent
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent

MAX_STEPS = 25
NUM_EPISODES = 2
SEED = 0


@pytest.fixture(params=["LunarLanderContinuous-v2"])
def continuous_environment(request):
    return request.param


@pytest.fixture(params=[BPTTAgent, SVGAgent])
def agent(request):
    return request.param


def rollout_agent(environment, agent):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(environment, SEED)
    agent = agent.default(environment, num_iter=2, num_epochs=2)
    train_agent(
        agent,
        environment,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        plot_flag=False,
    )
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)
    agent.logger.delete_directory()  # Cleanup directory.


def test_continuous_agent(continuous_environment, agent):
    rollout_agent(continuous_environment, agent)
