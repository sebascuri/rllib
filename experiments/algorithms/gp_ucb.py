"""Python Script Template."""

import gpytorch
import matplotlib.pyplot as plt
import torch

from rllib.agent.gp_ucb_agent import GPUCBAgent
from rllib.environment.bandit_environment import BanditEnvironment
from rllib.reward.gp_reward import GPBanditReward
from rllib.util import rollout_agent
from rllib.util.gaussian_processes import ExactGP, plot_gp


def plot(agent, objective, axis=None, noise=False):
    if axis is None:
        axis = plt.gca()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = agent.policy.x
        pred = agent.policy.gp(test_x)
        true_values = objective(None, test_x)[0].numpy()

        test_x = test_x.numpy()
        mean = pred.mean.numpy()
        error = 2 * pred.stddev.numpy()

        # Plot gp prediction
        axis.fill_between(test_x, mean - error, mean + error, lw=0, alpha=0.4,
                          color='C0')
        axis.plot(test_x, mean, color='C0')

        # Plot ground-truth
        axis.plot(test_x, true_values, '--', color='k')

        # Plot data
        axis.plot(agent.policy.gp.train_inputs[0].numpy(),
                  agent.policy.gp.train_targets.numpy(),
                  'x', markeredgewidth=2, markersize=5, color='C1')

        axis.set_xlim(test_x[0], test_x[-1])
        axis.set_ylim(-2.1, 2.1)
        axis.set_xticks([])
        axis.set_yticks([])
        # axis.set_xlabel('Inputs')
        # axis.set_ylabel('Objective')


if __name__ == '__main__':
    NUM_POINTS = 1000
    STEPS = 10
    SEED = 42
    torch.manual_seed(SEED)

    # Define objective function
    X = torch.tensor([-1., 1., 2.5, 4., 6])
    Y = 2 * torch.tensor([-0.5, 0.3, -0.2, .6, -0.5])

    x = torch.linspace(-1, 6, NUM_POINTS)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.noise = 0.1 ** 2
    objective_function = ExactGP(X, Y, likelihood)
    objective_function.eval()
    objective = GPBanditReward(objective_function)

    plt.plot(X.numpy(), Y.numpy(), '*')
    plot_gp(x, objective_function, num_samples=0)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.ion()
    plt.show()

    x0 = x[x > 0.2][[0]]
    y0 = objective(None, x0)[0].type(torch.get_default_dtype())
    model = ExactGP(x0, y0, likelihood)
    model.covar_module.base_kernel.lengthscale = 1
    agent = GPUCBAgent(model, x, beta=2.0)
    environment = BanditEnvironment(objective,
                                    x_min=x[[0]].numpy(), x_max=x[[-1]].numpy())
    state = environment.reset()

    fig, axes = plt.subplots(5, 2)
    for i in range(STEPS):
        plot(agent, objective, axis=axes[i % 5][i // 5])
        rollout_agent(environment, agent, num_episodes=1, max_steps=1)

    plt.ioff()
    plt.show()
