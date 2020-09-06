"""Implementation of a Logger class."""

import json
import os
import shutil
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter


class Logger(object):
    """Class that implements a logger of statistics.

    Parameters
    ----------
    name: str
        Name of logger. This create a folder at runs/`name'.
    comment: str, optional.
        This is useful to separate equivalent runs.
        The folder is runs/`name'/`comment_date'.
    tensorboard: bool, optional.
        Flag that indicates whether or not to save the results in the tensorboard.
    """

    def __init__(self, name, comment="", tensorboard=False):
        self.statistics = list()
        self.current = dict()
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        comment = comment + "_" + current_time if len(comment) else current_time
        self.writer = SummaryWriter(f"runs/{name}/{comment}")
        self._tensorboard = tensorboard

        self.episode = 0
        self.keys = set()

        if not self._tensorboard:
            for file in filter(
                lambda x: x.startswith("events"), os.listdir(self.writer.logdir)
            ):
                os.remove(f"{self.writer.logdir}/{file}")

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
        for key in sorted(self.keys):
            values = self.get(key)
            key = " ".join(key.split("_")).title()
            str_ += f"{key} Last: {values[-1]:.2g}. "
            str_ += f"Average: {np.mean(values):.2g}. "
            str_ += f"10-Episode: {np.mean(values[-10:]):.2g}.\n"

        return str_

    def get(self, key):
        """Return the statistics of a specific key."""
        return [statistic[key] for statistic in self.statistics if key in statistic]

    def update(self, **kwargs):
        """Update the statistics for the current episode.

        Parameters
        ----------
        kwargs: dict
            Any kwargs passed to update is converted to numpy and averaged
            over the course of an episode.
        """
        for key, value in kwargs.items():
            self.keys.add(key)
            if isinstance(value, torch.Tensor):
                value = value.detach().numpy()
            value = np.nan_to_num(value)
            if isinstance(value, np.float32):
                value = float(value)
            if isinstance(value, np.int64):
                value = int(value)

            if key not in self.current:
                self.current[key] = (1, value)
            else:
                count, old_value = self.current[key]
                new_count = count + 1
                new_value = old_value + (value - old_value) * (1 / new_count)
                self.current[key] = (new_count, new_value)

            if self._tensorboard:
                self.writer.add_scalar(
                    f"episode_{self.episode}/{key}",
                    self.current[key][1],
                    global_step=self.current[key][0],
                )

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
            self.keys.add(key)
            if isinstance(value, float) or isinstance(value, int) and self._tensorboard:
                self.writer.add_scalar(
                    f"average/{key}", value, global_step=self.episode
                )

        self.statistics.append(data)
        self.current = dict()
        self.episode += 1

    def save_hparams(self, hparams):
        """Save hparams to a json file."""
        with open(f"{self.writer.logdir}/hparams.json", "w") as f:
            json.dump(hparams, f)

    def export_to_json(self):
        """Save the statistics to a json file."""
        with open(f"{self.writer.logdir}/statistics.json", "w") as f:
            json.dump(self.statistics, f)

    def log_hparams(self, hparams, metrics=None):
        """Log hyper parameters together with a metric dictionary."""
        if not self._tensorboard:  # Do not save.
            return
        for k, v in hparams.items():
            if v is None:
                hparams[k] = 0
        self.writer.add_hparams(
            hparam_dict=hparams, metric_dict=metrics, name="hparams", global_step=1
        )

    def delete_directory(self):
        """Delete writer directory.

        Notes
        -----
        Use with caution. This will erase the directory, not the object.
        """
        shutil.rmtree(self.writer.logdir)
