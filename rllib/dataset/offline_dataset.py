"""An Offline dataset is intended for an offline rl algorithm to use."""

from dataclasses import asdict

import numpy as np
import torch
from torch.distributions import Poisson
from torch.utils.data.dataset import Dataset

from .datatypes import Observation


class OfflineDataset(Dataset):
    """Offline dataset that handles the get item."""

    def __init__(
        self,
        dataset,
        transformations=(),
        num_bootstraps=1,
        bootstrap=True,
        init_transformations=True,
    ):

        size, num_steps = dataset.state.shape[:2]
        self.num_steps = num_steps
        self.indexes = torch.arange(size)

        self.dataset = dataset

        self.transformations = transformations
        self.bootstrap = bootstrap
        if self.bootstrap:
            self.weights = Poisson(torch.ones(num_bootstraps)).sample((len(self),))
        else:
            self.weights = torch.ones(len(self), num_bootstraps)

        if init_transformations:
            self.init_transformations()
        self.max_len = len(self)

    @property
    def num_bootstraps(self):
        """Get the number of bootstraps."""
        return self.weights.shape[-1]

    @num_bootstraps.setter
    def num_bootstraps(self, num_bootstraps):
        """Set the number of bootstraps."""
        if self.bootstrap:
            self.weights = Poisson(torch.ones(num_bootstraps)).sample((len(self),))
        else:
            self.weights = torch.ones(len(self), num_bootstraps)

    def __iter__(self):
        """Iterate through data set."""
        for idx in range(len(self)):
            yield self._get_observation(idx)

    def __len__(self):
        """Length of data set."""
        return len(self.indexes)

    def __getitem__(self, idx):
        """Get item of dataset."""
        return asdict(self._get_observation(idx)), idx, self.weights[idx]

    def init_transformations(self):
        """Initialize transformations."""
        observation = self.dataset.clone()
        for transformation in self.transformations:
            transformation.update(observation)
            observation = transformation(observation.clone())

    def apply_transformations(self, observation):
        """Apply transformations to observation."""
        for transform in self.transformations:
            observation = transform(observation)
        return observation

    def get_random_split(self, ratio):
        """Get a random split of the dataset."""
        length = len(self)
        indexes = np.arange(length)
        np.random.shuffle(indexes)
        split = int(ratio * length)
        train_idx, validation_idx = indexes[:split], indexes[split:]

        train_set = OfflineDataset(
            self._get_raw_observation(train_idx),
            transformations=self.transformations,
            num_bootstraps=self.num_bootstraps,
            bootstrap=self.bootstrap,
            init_transformations=False,
        )
        validation_set = OfflineDataset(
            self._get_raw_observation(validation_idx),
            transformations=self.transformations,
            num_bootstraps=self.num_bootstraps,
            bootstrap=self.bootstrap,
            init_transformations=False,
        )

        return train_set, validation_set

    def _get_raw_observation(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: np.ndarray or int

        Returns
        -------
        observation: Observation
        """
        observation = Observation(
            state=self.dataset.state[idx],
            action=self.dataset.action[idx],
            reward=self.dataset.reward[idx],
            next_state=self.dataset.next_state[idx],
            done=self.dataset.done[idx],
            log_prob_action=self.dataset.log_prob_action[idx],
        ).clone()
        return observation

    def _get_observation(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: np.ndarray or int

        Returns
        -------
        observation: Observation
        """
        return self.apply_transformations(self._get_raw_observation(idx))

    def sample_batch(self, batch_size):
        """Sample a batch of data with a given size."""
        idx = np.random.choice(len(self), batch_size)
        return self._get_observation(idx), idx, self.weights[idx]

    @property
    def all_data(self):
        """Get all the transformed data."""
        return self.apply_transformations(self.all_raw)

    @property
    def all_raw(self):
        """Get all the un-transformed data."""
        return self.dataset.clone()
