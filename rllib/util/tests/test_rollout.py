from rllib.util import rollout_policy, rollout_agent
from rllib.policy import RandomPolicy
from rllib.agent import RandomAgent
from rllib.environment import GymEnvironment
import pytest


@pytest.fixture(params=['CartPole-v0', 'Pendulum-v0'])
def environment(request):
    return GymEnvironment(request.param)


def test_rollout_policy(environment):
    policy = RandomPolicy(environment.dim_state, environment.dim_action,
                          num_actions=environment.num_actions)
    trajectory = rollout_policy(environment, policy)

    assert len(trajectory) > 0


def test_rollout_agent(environment):
    agent = RandomAgent(environment.dim_state, environment.dim_action,
                        num_actions=environment.num_actions)
    rollout_agent(environment, agent)
