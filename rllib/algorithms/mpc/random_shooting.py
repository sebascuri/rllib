"""MPC Algorithms."""
from .cem_shooting import CEMShooting


class RandomShooting(CEMShooting):
    r"""Random Shooting solves the MPC problem by random sampling.

    The sampling distribution is a Multivariate Gaussian and the average of the best
    `num_elites' samples (action sequences) is returned.

    In practice, this is just is the first step of the Cross Entropy Method (n_iter=1).

    Parameters
    ----------
    dynamical_model: state transition model.
    reward_model: reward model.
    horizon: int.
        Horizon to solve planning problem.
    gamma: float, optional.
        Discount factor.
    num_samples: int, optional.
        Number of samples for shooting method.
    num_elites: int, optional.
        Number of elite samples to keep between iterations.
    termination: Callable, optional.
        Termination condition.
    terminal_reward: terminal reward model, optional.
    warm_start: bool, optional.
        Whether or not to start the optimization with a warm start.
    default_action: str, optional.
         Default action behavior.

    References
    ----------
    Nagabandi, A., Kahn, G., Fearing, R. S., & Levine, S. (2018).
    Neural network dynamics for model-based deep reinforcement learning with model-free
    fine-tuning. ICRA.

    Richards, A. G. (2005).
    Robust constrained model predictive control. Ph. D.

    Rao, A. V. (2009).
    A survey of numerical methods for optimal control.
    Advances in the Astronautical Sciences.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(num_iter=kwargs.pop("num_iter", 1), *args, **kwargs)
