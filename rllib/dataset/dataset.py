import numpy as np
from torch.utils import data
from . import Observation
from rllib.util.utilities import stack_list_of_tuples


class TrajectoryDataset(data.Dataset):
    def __init__(self, sequence_length=1, transforms=None):
        super().__init__()
        self._sequence_length = sequence_length
        self._trajectories = []
        self._sub_trajectories = []
        if transforms is None:
            transforms = []
        self._transforms = transforms

    def __getitem__(self, idx):
        trajectory_idx, start = self._sub_trajectories[idx]
        end = start + self._sequence_length

        trajectory = self._trajectories[trajectory_idx]

        observation = Observation(
            state=trajectory.state[start:end],
            action=trajectory.action[start:end],
            reward=trajectory.reward[start:end],
            next_state=trajectory.next_state[start:end],
            done=trajectory.done[start:end]
        )
        for transform in self._transforms:
            observation = transform(observation)

        return observation

    def __len__(self):
        return len(self._sub_trajectories)

    @property
    def initial_states(self):
        """Return a list with initial states."""
        return np.stack([trajectory.state[0] for trajectory in self._trajectories])

    def append(self, trajectory):
        num_observations = len(trajectory)
        trajectory_idx = len(self._trajectories)

        trajectory = stack_list_of_tuples(trajectory, dtype=np.float32)
        self._trajectories.append(trajectory)
        for transform in self._transforms:
            transform.update(trajectory)

        for i in range(num_observations // self._sequence_length):
            new_idx = (trajectory_idx, i * self._sequence_length)
            self._sub_trajectories.append(new_idx)

        # Append the final data batch
        if num_observations % self._sequence_length > 0:
            new_idx = (trajectory_idx, num_observations - self._sequence_length)
            self._sub_trajectories.append(new_idx)

    def shuffle(self):
        np.random.shuffle(self._sub_trajectories)
