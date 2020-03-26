import gpytorch
import pytest
import torch

from rllib.agent.gp_ucb_agent import GPUCBAgent
from rllib.environment.bandit_environment import BanditEnvironment
from rllib.reward.gp_reward import GPBanditReward
from rllib.util import rollout_agent
from rllib.util.gaussian_processes import ExactGP

NUM_POINTS = 1000
STEPS = 10
SEED = 42


@pytest.fixture
def reward():
    X = torch.tensor([-1., 1., 2.5, 4., 6])
    Y = 2 * torch.tensor([-0.5, 0.3, -0.2, .6, -0.5])

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.noise = 0.1 ** 2
    objective_function = ExactGP(X, Y, likelihood)
    objective_function.eval()

    return GPBanditReward(objective_function)


def test_GPUCB(reward):
    torch.manual_seed(SEED)
    x = torch.linspace(-1, 6, NUM_POINTS)
    x0 = x[x > 0.2][[0]]
    y0 = reward(None, x0)[0]
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.noise = 0.1 ** 2
    model = ExactGP(x0, y0, likelihood)
    environment = BanditEnvironment(reward,
                                    x_min=x[[0]].numpy(), x_max=x[[-1]].numpy())
    agent = GPUCBAgent(environment.name, model, x, beta=2.0)

    rollout_agent(environment, agent, num_episodes=1, max_steps=STEPS)
