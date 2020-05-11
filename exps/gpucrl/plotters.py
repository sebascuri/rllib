"""Plotters for GP-UCRL experiments."""
import os
import io

import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

from rllib.dataset.utilities import stack_list_of_tuples


def plot_last_trajectory(agent, episode: int):
    """Plot agent last trajectory."""
    model = agent.dynamical_model.base_model
    real_trajectory = stack_list_of_tuples(agent.last_trajectory)

    for transformation in agent.dataset.transformations:
        real_trajectory = transformation(real_trajectory)

    fig, axes = plt.subplots(model.dim_state + model.dim_action + 1, 1, sharex='col')

    for i in range(model.dim_state):
        axes[i].plot(real_trajectory.state[:, i])
        axes[i].set_ylabel(f"State {i}")

    for i in range(model.dim_action):
        axes[model.dim_state + i].plot(real_trajectory.action[:, i])
        axes[model.dim_state + i].set_ylabel(f"Action {i}")

    axes[-1].plot(real_trajectory.reward)
    axes[-1].set_ylabel(f"Reward {i}")
    axes[-1].set_xlabel('Time')

    img_name = f"{agent.comment.title()}"
    plt.suptitle(f'{img_name} Episode {episode + 1}', y=1)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    agent.logger.writer.add_image(img_name, image, episode)

    if 'DISPLAY' in os.environ:
        plt.show()
    else:
        plt.savefig(f"{agent.logger.writer.logdir}/{episode + 1}.png")


def plot_last_rewards(agent, episode: int):
    """Plot agent last trajectory."""
    real_trajectory = stack_list_of_tuples(agent.last_trajectory)

    for transformation in agent.dataset.transformations:
        real_trajectory = transformation(real_trajectory)

    fig, axes = plt.subplots(1, 1, sharex='col')
    axes.plot(real_trajectory.reward)
    axes.set_ylabel(f"Reward")
    axes.set_xlabel('Time')

    img_name = f"{agent.comment.title()}"
    plt.suptitle(f'{img_name} Episode {episode + 1}', y=1)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    agent.logger.writer.add_image(img_name, image, episode)

    if 'DISPLAY' in os.environ:
        plt.show()
    else:
        plt.savefig(f"{agent.logger.writer.logdir}/{episode + 1}.png")

