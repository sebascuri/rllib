# from rllib.policy import NNPolicy
# import torch
# from torch.distributions import Normal, Categorical
# import pytest
#
#
# @pytest.fixture(params=[(True, 4, 2), (False, 4, 4)])
# def nn_policy(request):
#     discrete = request.param[0]
#     state_dim = request.param[1]
#     action_dim = request.param[2]
#     if discrete:
#         action_space = Categorical(torch.ones((action_dim,)))
#     else:
#         action_space = Normal(torch.zeros((action_dim,)), torch.ones((action_dim,)))
#
#     return NNPolicy(action_space, state_dim), state_dim, action_dim
#
#
# def test_init(nn_policy):
#     pass
#
#
# def test_random_action(nn_policy):
#     nn_policy, state_dim, action_dim = nn_policy
#     distribution = nn_policy.random_action()
#     sample = distribution.sample()
#     if distribution.has_enumerate_support:  # Discrete
#         assert distribution.logits.shape == (action_dim,)
#         assert sample.shape == ()
#     else:  # Continuous
#         assert distribution.mean.shape == (action_dim,)
#         assert sample.shape == (action_dim,)
#
#
# def test_forward(nn_policy):
#     nn_policy, state_dim, action_dim = nn_policy
#     state = torch.randn(state_dim, )
#     distribution = nn_policy(state)
#     sample = distribution.sample()
#
#     if distribution.has_enumerate_support:  # Discrete
#         assert distribution.logits.shape == (action_dim,)
#         assert sample.shape == ()
#     else:  # Continuous
#         assert distribution.mean.shape == (action_dim,)
#         assert sample.shape == (action_dim,)
#
#
# def test_batch(nn_policy):
#     batch_size = 32
#     nn_policy, state_dim, action_dim = nn_policy
#     state = torch.randn(batch_size, state_dim, )
#     distribution = nn_policy.action(state)
#     sample = distribution.sample()
#
#     if distribution.has_enumerate_support:  # Discrete
#         assert distribution.logits.shape == (batch_size, action_dim,)
#         assert sample.shape == (batch_size,)
#     else:  # Continuous
#         assert distribution.mean.shape == (batch_size, action_dim,)
#         assert sample.shape == (batch_size, action_dim,)
#
#
# def test_sanity_distribution_check(nn_policy):
#     nn_policy, state_dim, action_dim = nn_policy
#     state = torch.randn(state_dim, )
#     distribution = nn_policy.action(state)
#     random_distribution = nn_policy.random()
#     assert distribution.event_shape == random_distribution.event_shape
#     assert distribution.batch_shape == random_distribution.batch_shape
