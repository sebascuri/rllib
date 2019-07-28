from rllib.util import rollout_policy, rollout_agent
from rllib.policy import RandomPolicy
from rllib.agent import RandomAgent
import pytest
import gym


@pytest.fixture(params=['CartPole-v0', 'Pendulum-v0'])
def environment(request):
    env = gym.make(request.param)
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, 'n'):
        num_action = env.action_space.n
        dim_action = 1
    else:
        num_action = None
        dim_action = env.action_space.shape[0]

    return env, state_dim, dim_action, num_action


def test_rollout_policy(environment):
    env, state_dim, dim_action, num_action = environment
    policy = RandomPolicy(state_dim, dim_action, num_action)
    trajectory = rollout_policy(env, policy)

    assert len(trajectory) > 0


def test_rollout_agent(environment):
    env, state_dim, dim_action, num_action = environment
    agent = RandomAgent(state_dim, dim_action, num_action)
    rollout_agent(env, agent)
