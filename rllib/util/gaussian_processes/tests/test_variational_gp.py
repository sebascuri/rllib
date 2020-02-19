# from rllib.util.gaussian_processes import VariationalGP
# import torch
# import pytest
#
#
# @pytest.fixture(params=[(100, 4)])
# def gp(request):
#     num_inducing_points = request.param[0]
#     in_dim = request.param[1]
#     inducing_points = torch.randn((num_inducing_points, in_dim))
#     gp_ = VariationalGP(inducing_points)
#     return gp_, in_dim
#
#
# def test_init(gp):
#     pass
#
#
# def test_forward(gp):
#     gp, in_dim = gp
#     batch_size = 32
#
#     tensor = torch.rand((batch_size, in_dim))
#     distribution = gp.forward(tensor)
#
#     assert distribution.mean.shape == (batch_size, )
#     assert distribution.has_rsample
#     assert not distribution.has_enumerate_support
