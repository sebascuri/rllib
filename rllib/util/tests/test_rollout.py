import pytest

from rllib.agent import RandomAgent
from rllib.environment import GymEnvironment
from rllib.environment.mdps import EasyGridWorld
from rllib.policy import RandomPolicy
from rllib.util.rollout import rollout_agent, rollout_policy


@pytest.fixture(
    params=["CartPole-v0", "Pendulum-v0", "MountainCarContinuous-v0", "Taxi-v3"]
)
def environment(request):
    return GymEnvironment(request.param)


def test_rollout_policy(environment):
    policy = RandomPolicy(
        environment.dim_state,
        environment.dim_action,
        num_actions=environment.num_actions,
    )
    trajectory = rollout_policy(environment, policy)

    assert len(trajectory) > 0


def test_rollout_agent(environment):
    agent = RandomAgent.default(environment)
    rollout_agent(environment, agent)


def test_rollout_easy_grid_world():
    environment = EasyGridWorld()
    agent = RandomAgent.default(environment)
    rollout_agent(environment, agent, max_steps=20)

    policy = agent.policy
    rollout_policy(environment, policy)
