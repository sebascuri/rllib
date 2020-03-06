"""Implementation of a Logger class."""

import numpy as np


class Logger(object):
    """Class that implements the decay of a parameter.

    It has two properties:
    running_log:
        list of all the values that the log holds.
    episode_log:
        list of all the values, averaged per episode, that the log holds.

    Parameters
    ----------
    type_: str
        string that identifies the logger type.
        - sum: stores the sum of the values encounter in the episode..
        - mean: stores the average of the values encounter in the episode.
        - max: stores the maximum of the values encounter in the episode..
        - min: stores the minimum of the values encounter in the episode..
        - abs_mean: stores the average of the absolute values encounter in the episode.
        - abs_sum: stores the maximum of the absolute values encounter in the episode.
        - abs_max: stores the minimum of the absolute values encounter in the episode.
    """

    def __init__(self, type_='sum'):
        self.running_log = []
        self.episode_log = []
        self.type_ = type_
        self.current_log = []

    def start_episode(self):
        """Start a new episode."""
        self.current_log = []

    def append(self, value):
        """Append a new value.

        Parameters
        ----------
        value: float or int.
            new value to append.
        """
        self.running_log.append(value)
        self.current_log.append(value)

    def end_episode(self):
        """End an episode."""
        if len(self.current_log) > 0:
            log = np.array(self.current_log)
            if self.type_ == 'sum':
                self.episode_log.append(log.sum())
            elif self.type_ == 'mean':
                self.episode_log.append(log.mean())
            elif self.type_ == 'max':
                self.episode_log.append(log.max())
            elif self.type_ == 'min':
                self.episode_log.append(log.min())
            elif self.type_ == 'abs_sum':
                self.episode_log.append(np.abs(log).sum())
            elif self.type_ == 'abs_mean':
                self.episode_log.append(np.abs(log).mean())
            elif self.type == 'abs_max':
                self.episode_log.append(np.abs(log).max())

    def dump(self, name):
        """Save the logs. TODO: Implement it."""
        pass
