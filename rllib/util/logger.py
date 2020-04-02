"""Implementation of a Logger class."""

from datetime import datetime

import json
import numpy as np
import torch
from tensorboardX import SummaryWriter


class Logger(object):
    """Class that implements a logger of statistics.

    Parameters
    ----------
    name: str
    """

    def __init__(self, name, comment=''):
        self.statistics = list()
        self.current = dict()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(
            f"runs/{name}/{comment + '_' + current_time if comment else current_time}"
        )

    def __len__(self):
        """Return the number of episodes."""
        return len(self.statistics)

    def __iter__(self):
        """Iterate over the episode statistics."""
        return self.statistics

    def __getitem__(self, index):
        """Return a specific episode."""
        return self.statistics[index]

    def __str__(self):
        """Return parameter string of logger."""
        str_ = ""
        for key in self.keys():
            values = self.get(key)
            key = ' '.join(key.split('_')).title()
            str_ += f"{key} Last: {values[-1]:.2g}. "
            str_ += f"Average: {np.mean(values):.2g}. "
            str_ += f"10-Episode: {np.mean(values[-10:]):.2g}.\n"

        return str_

    def get(self, key):
        """Return the statistics of a specific key."""
        return [statistic[key] for statistic in self.statistics if key in statistic]

    def keys(self):
        """Return iterator of stored keys."""
        if len(self.statistics):
            return self.statistics[-1].keys()
        else:
            return []

    def update(self, **kwargs):
        """Update the statistics for the current episode.

        Parameters
        ----------
        kwargs: dict
            Any kwargs passed to update is converted to numpy and averaged
            over the course of an episode.
        """
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().numpy()
            value = np.nan_to_num(value)
            if key not in self.current:
                self.current[key] = (1, value)
            else:
                count, old_value = self.current[key]
                new_count = count + 1
                new_value = old_value + (value - old_value) * (1 / new_count)
                self.current[key] = (new_count, new_value)

            self.writer.add_scalar(f"current/{key}", self.current[key][1])

    def end_episode(self, **kwargs):
        """Finalize collected data and add final fixed values.

        Parameters
        ----------
        kwargs : dict
            Any kwargs passed to end_episode overwrites tracked data if present.
            This can be used to store fixed values that are tracked per episode
            and do not need to be averaged.
        """
        data = {key: value[1] for key, value in self.current.items()}
        kwargs = {key: value for key, value in kwargs.items()}
        data.update(kwargs)

        for key, value in data.items():
            if isinstance(value, float) or isinstance(value, int):
                self.writer.add_scalar(f"episode/{key}", value)

        self.statistics.append(data)
        self.current = dict()

    def export_to_json(self, hparams=None):
        """Save the statistics (and hparams) to a json file."""
        with open(f"{self.writer.logdir}/statistics.json", "w") as f:
            json.dump(self.statistics, f)
        if hparams is not None:
            with open(f"{self.writer.logdir}/hparams.json", "w") as f:
                json.dump(hparams, f)
