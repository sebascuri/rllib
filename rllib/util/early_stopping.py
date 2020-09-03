"""Early Stopping algorithm."""
from rllib.util.utilities import MovingAverage


class EarlyStopping(object):
    """Early stopping algorithm.

    Parameters
    ----------
    epsilon: float.
        Parameter that controls the early stopping.
        If epsilon < 0 it will never raise the stop flag.
        If the algorithm did not start tracking, it will not raise the stop flag.

    relative: bool.
        Flag that indicates whether the stopping condition is relative or absolute.
        When relative, the stop flag is true when average > (1+epsilon) * minimum.
        When absolute, the stop flag is true when average > epsilon.
    """

    def __init__(self, epsilon, relative=True):
        self.epsilon = epsilon
        self.relative = relative
        self.moving_average = []
        self.min_value = []

    @property
    def stop(self):
        """Return true if it is ready to early stop."""
        if self.min_value is None:
            return False
        if self.epsilon < 0:
            return False

        for i, min_value in enumerate(self.min_value):
            moving_average = self.moving_average[i].value

            if self.relative:
                if moving_average > (1 + self.epsilon) * min_value:
                    return True
            else:
                if moving_average > self.epsilon:
                    return True

            if moving_average < min_value:
                self.min_value[i] = moving_average

        return False

    def _reset(self, num, hard):
        self.moving_average = [MovingAverage() for _ in range(num)]

        if len(self.min_value) < num or hard:
            self.min_value = [float("inf") for _ in range(num)]

    def reset(self, hard=True):
        """Reset moving average and minimum values.

        Parameters
        ----------
        hard: bool, optional (default=True).
            If true, reset moving average and min_value.
            If false, reset only moving average.
        """
        self._reset(num=len(self.moving_average), hard=hard)

    def update(self, *args):
        """Update values."""
        if not len(self.moving_average):
            self._reset(num=len(args), hard=True)

        for arg, moving_average in zip(args, self.moving_average):
            moving_average.update(arg)
