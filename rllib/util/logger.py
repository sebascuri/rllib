"""Implementation of a Logger class."""

import time

import numpy as np
from tensorboardX import SummaryWriter


class Logger(object):
    """Class that implements a logger of statistics.

    Parameters
    ----------
    name: str
    """

    def __init__(self, name=''):
        self.statistics = list()
        self.current = dict()
        self.writer = SummaryWriter()
        self.name = name
        self.start = time.time()

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
            str_ += f"Average {key} {np.mean(values):.1f}\n"
            str_ += f"10-Episode {key} {np.mean(values[-10:]):.1f}\n"

        return str_

    def get(self, key):
        """Return the statistics of a specific key."""
        return [statistic[key] for statistic in self.statistics if key in statistic]

    def keys(self):
        """Return iterator of stored keys."""
        return self.statistics[-1].keys()

    def update(self, **kwargs):
        """Update the statistics for the current episode.

        Parameters
        ----------
        kwargs: dict
            Any kwargs passed to update is converted to numpy and averaged
            over the course of an episode.
        """
        for key, value in kwargs.items():
            value = np.nan_to_num(value)
            if key not in self.current:
                self.current[key] = (1, value)
            else:
                count, old_value = self.current[key]
                new_count = count + 1
                new_value = old_value + (value - old_value) * (1 / new_count)
                self.current[key] = (new_count, new_value)

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

        for key, value in self.current.items():
            self.writer.add_scalar(f"{self.name}/{key}", value[1], len(self),
                                   walltime=time.time() - self.start)

        self.statistics.append(data)
        self.current = dict()

    def dump(self, name):
        """Save the logs. TODO: Implement it."""
        pass
