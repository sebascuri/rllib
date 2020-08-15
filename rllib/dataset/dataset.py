"""Implementation of a Trajectory Dataset."""
from dataclasses import asdict

import numpy as np
import torch
from torch.utils import data

from rllib.dataset.datatypes import Observation

from .utilities import map_and_cast, stack_list_of_tuples


class TrajectoryDataset(data.Dataset):
    """A dataset has trajectories from different interactions.

    The dataset splits the dataset into subsequences of fixed length.

    Properties
    ----------
    sequence_length: int, optional(default: 1)
        sequence length of sub-trajectories
    transformations: list of transforms.AbstractTransform
        A sequence of transformations to apply to the dataset, each of which is a
        callable that takes an observation as input and returns a modified observation.
        If they have an `update` method it will be called whenever a new trajectory
        is added to the dataset.

    Methods
    -------
    initial_states:
        return list of initial states of all the trajectories.
    append(trajectory):
        append a trajectory to the dataset.
    shuffle():
        shuffle the dataset.
    sequence_length: int
        length of the sub-trajectories.

    """

    def __init__(self, sequence_length=None, transformations=None):
        super().__init__()
        self._sequence_length = sequence_length
        self._trajectories = []
        self._sub_trajectory_indexes = []
        self._num_points = 0
        self.transformations = transformations if transformations else []

    def __getitem__(self, idx):
        """Return any desired sub-trajectory.

        Parameters
        ----------
        idx: int

        Returns
        -------
        sub-trajectory: Observation
        """
        if self.sequence_length is None:  # get trajectory
            observation = self._trajectories[idx]
        else:  # Get sub-trajectory.
            trajectory_idx, start = self._sub_trajectory_indexes[idx]
            end = start + self._sequence_length

            trajectory = self._trajectories[trajectory_idx]

            observation = Observation(
                **{key: val[start:end] for key, val in asdict(trajectory).items()}
            )

        for transform in self.transformations:
            observation = transform(observation)

        return observation

    def __len__(self):
        """Return the size in the dataset.

        The size is the number of sub-trajectories. If the sequence length is None,
        then the size is the number of trajectories.

        Returns
        -------
        length: int

        """
        if self._sequence_length is None:
            return len(self._trajectories)
        else:
            return len(self._sub_trajectory_indexes)

    def append(self, trajectory):
        """Append new trajectories to the dataset.

        Parameters
        ----------
        trajectory: sized

        Raises
        ------
        ValueError
            If the new trajectory is shorter than the sequence length.
        """
        trajectory_index = len(self._trajectories)

        if isinstance(trajectory, Observation):
            try:
                num_observations = len(trajectory.reward)
            except TypeError:
                num_observations = 1
        else:
            # Stack the tuples to one trajectory
            num_observations = len(trajectory)
            trajectory = stack_list_of_tuples(trajectory)

        if (
            self._sequence_length is not None
            and num_observations < self._sequence_length
        ):
            raise ValueError("The sequence is shorter than the sequence length")

        # Add trajectory to dataset
        self._trajectories.append(trajectory)

        # Update the transformers but do not apply transforms.
        for transformation in self.transformations:
            transformation.update(trajectory)

        # Add sub-trajectory indexes
        if self._sequence_length is not None:
            sub_indexes = self._get_subindexes(num_observations, self._sequence_length)
            sub_indexes = [(trajectory_index, sub_index) for sub_index in sub_indexes]

            self._sub_trajectory_indexes += sub_indexes

    def shuffle(self):
        """Shuffle the dataset."""
        if self._sequence_length is not None:
            np.random.shuffle(self._trajectories)
        else:
            np.random.shuffle(self._sub_trajectory_indexes)

    @property
    def all_data(self):
        """Get all the data."""
        data = map_and_cast(torch.cat, self._trajectories)
        for transformation in self.transformations:
            data = transformation(data)
        return data

    @property
    def initial_states(self):
        """Return a list with initial states."""
        return np.stack([trajectory.state[0] for trajectory in self._trajectories])

    @property
    def sequence_length(self):
        """Return the sequence length of the sub-trajectories."""
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        """Set the sequence length and update the sub-trajectory indexes."""
        self._sequence_length = value
        self._sub_trajectory_indexes = []
        if value is not None:
            for trajectory_index, trajectory in enumerate(self._trajectories):
                num_observations = len(trajectory.state)
                sub_indexes = self._get_subindexes(num_observations, value)
                sub_indexes = [
                    (trajectory_index, sub_index) for sub_index in sub_indexes
                ]
                self._sub_trajectory_indexes += sub_indexes

    @staticmethod
    def _get_subindexes(num_observations, sequence_length, drop_last=False):
        """Extract indexes for sub-trajectories from sequence of observations.

        Parameters
        ----------
        num_observations : int
            The number of observations to split up
        sequence_length : int
            The length of the subsequences.
        drop_last : int
            Whether to drop the end of the sequence if num_observations is not divisible
            by the sequence length. If False, adds an extra subsequence that includes
            the final data points.

        Returns
        -------
        sub_indexes : list
            A list of integers that indicate the starting position of each subsequence.

        >>> from rllib.dataset import TrajectoryDataset
        >>> TrajectoryDataset._get_subindexes(10, 3)
        [0, 3, 6, 7]
        >>> TrajectoryDataset._get_subindexes(10, 3, drop_last=True)
        [0, 3, 6]
        >>> TrajectoryDataset._get_subindexes(9, 3)
        [0, 3, 6]
        """
        # Add sub-trajectory indexes
        num_sub_trajectories = num_observations // sequence_length
        sub_indexes = [i * sequence_length for i in range(num_sub_trajectories)]

        # In this case the last transitions are not included in the loop above.
        if not drop_last and num_observations % sequence_length > 0:
            # Append a subsequence that includes the last data point
            sub_indexes.append(num_observations - sequence_length)
        return sub_indexes
