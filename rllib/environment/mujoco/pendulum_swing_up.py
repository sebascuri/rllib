"""Pendulum Swing-up Environment with full observation."""
import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv


class PendulumSwingUpEnv(PendulumEnv):
    """Pendulum Swing-up Environment."""

    def reset(self):
        """Reset to fix initial conditions."""
        high = np.array([np.pi, 0])
        self.state = self.np_random.uniform(low=high, high=high)
        self.last_u = None
        return self._get_obs()
