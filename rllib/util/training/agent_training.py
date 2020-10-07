"""Python Script Template."""
import matplotlib.pyplot as plt
import numpy as np

from rllib.util.rollout import rollout_agent

from .utilities import Evaluate


def train_agent(agent, environment, plot_flag=True, *args, **kwargs):
    """Train an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    plot_flag: bool, optional.

    Other Parameters
    ----------------
    See rollout_agent.
    """
    agent.train()
    rollout_agent(environment, agent, *args, **kwargs)

    if plot_flag:
        for key in agent.logger.keys:
            plt.plot(agent.logger.get(key))
            plt.xlabel("Episode")
            plt.ylabel(" ".join(key.split("_")).title())
            plt.title(f"{agent.name} in {environment.name}")
            plt.show()
    print(agent)


def evaluate_agent(agent, environment, num_episodes, max_steps, render=True):
    """Evaluate an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    num_episodes: int
    max_steps: int
    render: bool
    """
    with Evaluate(agent):
        rollout_agent(
            environment,
            agent,
            max_steps=max_steps,
            num_episodes=num_episodes,
            render=render,
        )
        returns = np.mean(agent.logger.get("eval_return")[-num_episodes:])
        print(f"Test Cumulative Rewards: {returns}")
