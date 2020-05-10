"""Plotters for gp_ucrl acrobot experiments."""
import itertools
import os
import io

import matplotlib.pyplot as plt
import numpy as np
import torch
import PIL.Image
from torchvision.transforms import ToTensor

from rllib.model.gp_model import ExactGPModel
from rllib.util.utilities import moving_average_filter
from rllib.dataset.utilities import stack_list_of_tuples


def plot_cartpole_trajectories(agent, episode: int):
    """Plot GP inputs and trajectory in a Pendulum environment."""
    model = agent.mppo.dynamical_model.base_model
    real_trajectory = stack_list_of_tuples(agent.last_trajectory)
    sim_trajectory = agent.sim_trajectory

    for transformation in agent.dataset.transformations:
        real_trajectory = transformation(real_trajectory)
        sim_trajectory = transformation(sim_trajectory)

    fig, axes = plt.subplots(model.dim_state + model.dim_action + 1, 2,
                             sharex='col')

    for i in range(model.dim_state):
        axes[i, 0].plot(real_trajectory.state[:, i])
        axes[i, 1].plot(sim_trajectory.state[:, 0, :, i])
        axes[i, 0].set_ylabel(f"State {i}")

    for i in range(model.dim_action):
        axes[model.dim_state + i, 0].plot(real_trajectory.action[:, i])
        axes[model.dim_state + i, 1].plot(sim_trajectory.action[:, 0, :, i])
        axes[model.dim_state + i, 0].set_ylabel(f"Action {i}")

    axes[-1, 0].plot(real_trajectory.reward)
    axes[-1, 1].plot(sim_trajectory.reward[:, 0, :])

    for i in range(2):
        axes[-1, i].set_xlabel('Time')

    axes[0, 0].set_title('Real Trajectory')
    axes[0, 1].set_title('Simulated Trajectory')

    img_name = f"{agent.comment.title()}"
    plt.suptitle(f'{img_name} Episode {episode + 1}', y=1.0)

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
