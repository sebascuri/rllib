import numpy as np
from rllib.dataset import Observation


class Dataset(object):
    """A dataset is comprised of trajectories from different interactions.

    An observation is a tuple made of (state, action, reward, next_state).
    A trajectory is a sequence of observations.
    A dataset is a collection of such trajectories.

    One can iterate through a dataset with the __iter__ method.
    This method returns batches of size 'batch_size' of sub-trajectories of length
    'sequence_length'. Both 'batch_size' and 'sequence_length' are set with the
    'config' dictionary at initialization.

    A sub-trajectory is considered the unit data-point of the dataset.


    When a trajectory is added, it is split into sub-trajectories and these are added to
    the dataset. In the case some extra points don't complete a full sub-trajectory,
    these are stored in the dataset but they are not sampled. TODO: change this.


    The public methods are:
        append
        shuffle
        __len__
        __iter__
    """

    def __init__(self, state_dim, action_dim, batch_size=1, sequence_length=1,
                 seed=None):
        """Initialize dataset object.

        Parameters
        ----------
        state_dim: int
        action_dim: int
        batch_size: int, default = 1
        sequence_length: int, default = 1
        seed: int, optional

        """
        self._trajectory_indexes = []  # First points of the different trajectories.
        self._sub_trajectory_indexes = []  # First points of all sub-trajectories.

        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._state_dim = state_dim
        self._action_dim = action_dim

        self._num_points = 0
        self._num_trajectories = 0
        self._num_sub_trajectories = 0

        # Each new observation is in a new row.
        self._dataset = {
            'state': np.empty((0, self._state_dim)),
            'reward': np.empty((0, 1)),
            'action': np.empty((0, self._action_dim)),
            'next_state': np.empty((0, self._state_dim))
        }

        self.random = np.random.RandomState(seed)

    def __len__(self):
        """Number of sub-trajectories in the dataset.

        Returns
        -------
        length: int

        """
        return self.number_sub_trajectories

    def __iter__(self):
        """Dataset iterator that yields a batch of sub-trajectories

        Returns
        -------
        state: ndarray of size batch_size x sequence_length x state_dim
        action: ndarray of size batch_size x sequence_length x action_dim
        reward: ndarray of size batch_size x sequence_length x 1
        next_state: ndarray of size batch_size x sequence_length x state_dim

        """
        batch_size = self._batch_size
        sequence_length = self._sequence_length
        for i in range(len(self) // batch_size):
            indexes = self._sub_trajectory_indexes[batch_size * i:batch_size * (i + 1)]

            state = np.zeros(
                (batch_size, sequence_length, self._state_dim))
            action = np.zeros(
                (batch_size, sequence_length, self._action_dim))
            reward = np.zeros(
                (batch_size, sequence_length, 1))
            next_state = np.zeros(
                (batch_size, sequence_length, self._state_dim))

            for i_batch, idx in enumerate(indexes):
                state[i_batch] = self._dataset['state'][
                                 idx:(idx + sequence_length)]
                action[i_batch] = self._dataset['action'][
                                  idx:(idx + sequence_length)]
                reward[i_batch] = self._dataset['reward'][
                                  idx:(idx + sequence_length)]
                next_state[i_batch] = self._dataset['next_state'][
                                      idx:(idx + sequence_length)]

            yield state, action, reward, next_state

    def append(self, trajectory):
        """Append new trajectories to the dataset.

        Parameters
        ----------
        trajectory: list

        Returns
        -------
        None

        """
        assert len(trajectory) > 0
        observations = Observation(*[np.stack(x) for x in zip(*trajectory)])
        # observations = stack_trajectory(trajectory)

        sequence_length = self._sequence_length

        self._trajectory_indexes.append(self.number_observations)
        num_sub_trajectories = len(trajectory) // sequence_length
        for i in range(num_sub_trajectories):
            self._sub_trajectory_indexes.append(self.number_observations
                                                + i * sequence_length)

        for key, value in self._dataset.items():
            new_observation = getattr(observations, key)
            if new_observation.ndim == 1:
                new_observation = new_observation[:, None]
            self._dataset[key] = np.vstack((value, new_observation))

        #  When this condition is true, the last transitions will not get sampled.
        if len(trajectory) % sequence_length > 0:
            # TODO: Implement something for this cases (delayed rewards?).
            pass

        self._num_trajectories += 1
        self._num_sub_trajectories += num_sub_trajectories
        self._num_points += len(trajectory)

    def shuffle(self):
        """Shuffle the dataset.

        This method is useful to de-correlate the sub-trajectories.

        Returns
        -------
        None

        """
        self.random.shuffle(self._sub_trajectory_indexes)

    def split(self, train_split_ratio, shuffle=False):
        """
        Split the dataset into test and train Datasets.

        Parameters
        ----------
        train_split_ratio: float (0, 1)
            Fraction of sub-trajectories assigned to training set.
        shuffle: bool
            Shuffle the dataset before splitting.

        Returns
        -------
        train_dataset: Dataset
        test_dataset: Dataset

        """
        if shuffle:
            self.shuffle()
        split_index = train_split_ratio * len(self)
        train_idx = self._sub_trajectory_indexes[:int(split_index)]
        test_idx = self._sub_trajectory_indexes[int(split_index):]
        train = Dataset(self._state_dim, self._action_dim, self._batch_size,
                        self._sequence_length)
        test = Dataset(self._state_dim, self._action_dim, self._batch_size,
                       self._sequence_length)

        for idx, dataset in zip((test_idx, train_idx), (test, train)):
            dataset._num_trajectories = None
            dataset._num_points = len(idx) * self._sequence_length
            dataset._num_sub_trajectories = len(idx)

            dataset._dataset = self._dataset.copy()  # copy to loose reference.
            dataset._sub_trajectory_indexes = idx
        return train, test

    @property
    def number_trajectories(self):
        return self._num_trajectories

    @property
    def number_observations(self):
        return self._num_points

    @property
    def number_sub_trajectories(self):
        return self._num_sub_trajectories
