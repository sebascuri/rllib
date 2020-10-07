import numpy as np
import pytest
import torch

from rllib.agent import (
    A2CAgent,
    ActorCriticAgent,
    DDQNAgent,
    DPGAgent,
    DQNAgent,
    ExpectedActorCriticAgent,
    ExpectedSARSAAgent,
    GAACAgent,
    ISERAgent,
    MPCAgent,
    MPOAgent,
    PPOAgent,
    QLearningAgent,
    RandomAgent,
    REINFORCEAgent,
    REPSAgent,
    SACAgent,
    SARSAAgent,
    SoftQLearningAgent,
    SVG0Agent,
    TD3Agent,
    TRPOAgent,
    VMPOAgent,
)
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent

MAX_STEPS = 25
NUM_EPISODES = 6
SEED = 0


@pytest.fixture(params=["CartPole-v0", "NChain-v0"])
def discrete_environment(request):
    return request.param


@pytest.fixture(params=["LunarLanderContinuous-v2"])
def continuous_environment(request):
    return request.param


@pytest.fixture(
    params=[
        ActorCriticAgent,
        A2CAgent,
        ExpectedActorCriticAgent,
        GAACAgent,
        ISERAgent,
        REINFORCEAgent,
        ExpectedSARSAAgent,
        SARSAAgent,
        QLearningAgent,
        SoftQLearningAgent,
        DDQNAgent,
        DQNAgent,
        REPSAgent,
        MPOAgent,
        PPOAgent,
        TRPOAgent,
        RandomAgent,
        VMPOAgent,
    ]
)
def discrete_agent(request):
    return request.param


@pytest.fixture(
    params=[
        ActorCriticAgent,
        A2CAgent,
        ExpectedActorCriticAgent,
        GAACAgent,
        REINFORCEAgent,
        ISERAgent,
        DPGAgent,
        TD3Agent,
        SACAgent,
        REPSAgent,
        MPOAgent,
        MPCAgent,
        PPOAgent,
        SVG0Agent,
        TRPOAgent,
        RandomAgent,
        VMPOAgent,
    ]
)
def continuous_agent(request):
    return request.param


def rollout_agent(environment, agent):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    environment = GymEnvironment(environment, SEED)
    agent = agent.default(environment, num_iter=1, num_epochs=2)
    agent.num_rollouts = min(agent.num_rollouts, NUM_EPISODES // 6)
    train_agent(
        agent,
        environment,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        plot_flag=False,
    )
    evaluate_agent(agent, environment, 1, MAX_STEPS, render=False)
    agent.logger.delete_directory()  # Cleanup directory.


def test_discrete_agent(discrete_environment, discrete_agent):
    rollout_agent(discrete_environment, discrete_agent)


def test_continuous_agent(continuous_environment, continuous_agent):
    rollout_agent(continuous_environment, continuous_agent)
