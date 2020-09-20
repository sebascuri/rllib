"""Python Script Template."""
import matplotlib.pyplot as plt
import numpy as np

from rllib.util.rollout import rollout_agent

from .utilities import Evaluate


def train_agent(
    agent,
    environment,
    num_episodes,
    max_steps,
    plot_flag=True,
    print_frequency=0,
    eval_frequency=0,
    plot_frequency=0,
    save_milestones=None,
    render=False,
    plot_callbacks=None,
):
    """Train an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    num_episodes: int
    max_steps: int
    plot_flag: bool, optional.
    print_frequency: int, optional.
    eval_frequency: int, optional.
    plot_frequency: int
    save_milestones: List[int], optional.
        List with episodes in which to save the agent.
    render: bool, optional.
    plot_callbacks: list, optional.

    """
    agent.train()
    rollout_agent(
        environment,
        agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
        print_frequency=print_frequency,
        plot_frequency=plot_frequency,
        eval_frequency=eval_frequency,
        save_milestones=save_milestones,
        render=render,
        plot_callbacks=plot_callbacks,
    )

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
