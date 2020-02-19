import gpytorch
import torch
from rllib.agent.gp_ucb_agent import GPUCBAgent
from rllib.environment.bandit_environment import BanditEnvironment
from rllib.util import rollout_agent
from rllib.util.gaussian_processes import ExactGPModel
import pytest

NUM_POINTS = 1000
STEPS = 10
SEED = 42


@pytest.fixture
def objective():
    X = torch.tensor([-1., 1., 2.5, 4., 6])
    Y = 2 * torch.tensor([-0.5, 0.3, -0.2, .6, -0.5])

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.noise = 0.1 ** 2
    objective_function = ExactGPModel(X, Y, likelihood)
    objective_function.eval()

    def predict(x, model, noise=True):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if type(x) is not torch.Tensor:
                x = torch.tensor(x)
            if x.ndim == 0:
                x = x.unsqueeze(0)

            pred = model(x).mean
            if noise:
                pred = model.likelihood(pred).sample()
            return pred.numpy()

    return lambda x_, noise=True: predict(x_, objective_function, noise)


def test_GPUCB(objective):
    torch.manual_seed(SEED)
    x = torch.linspace(-1, 6, NUM_POINTS)
    x0 = x[x > 0.2][[0]]
    y0 = torch.tensor(objective(x0)).float()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.noise = 0.1 ** 2
    model = ExactGPModel(x0, y0, likelihood)
    agent = GPUCBAgent(model, x, beta=2.0)
    environment = BanditEnvironment(objective,
                                    x_min=x[[0]].numpy(), x_max=x[[-1]].numpy())

    rollout_agent(environment, agent, num_episodes=1, max_steps=STEPS)
