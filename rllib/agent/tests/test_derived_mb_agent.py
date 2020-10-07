import numpy as np
import pytest
import torch

from rllib.agent import DynaAgent, MVEAgent, STEVEAgent
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent

MAX_STEPS = 25
NUM_EPISODES = 4
SEED = 0


@pytest.fixture(params=["LunarLanderContinuous-v2"])
def continuous_environment(request):
    return request.param


@pytest.fixture(params=[1, 4])
def num_steps(request):
    return request.param


@pytest.fixture(params=["DPG", "TD3", "SAC", "MPO", "VMPO"])
def base_agent(request):
    return request.param


@pytest.fixture(params=[DynaAgent, STEVEAgent, MVEAgent])
def extender(request):
    return request.param


def rollout_agent(environment, base_agent, extender, num_steps, td_k=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(environment, SEED)
    agent = extender.default(
        environment,
        base_agent_name=base_agent,
        num_steps=num_steps,
        num_samples=2,
        num_iter=2,
        num_epochs=2,
        td_k=td_k,
    )
    train_agent(
        agent,
        environment,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        plot_flag=False,
    )
    evaluate_agent(
        agent, environment, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, render=False
    )
    agent.logger.delete_directory()  # Cleanup directory.


def test_continuous_agent(continuous_environment, base_agent, extender, num_steps):
    rollout_agent(continuous_environment, base_agent, extender, num_steps)


def test_mve_not_td_k(continuous_environment, base_agent, num_steps):
    rollout_agent(continuous_environment, base_agent, MVEAgent, num_steps, td_k=False)
