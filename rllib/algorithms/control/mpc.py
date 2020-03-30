"""LQR algorithms."""
import numpy as np
import scipy


def dlqr(a, b, q, r, gamma=None):
    """Solve the discrete time lqr controller.

    x[k+1] = a @ x[k] + b @ u[k]

    with instantaneous cost
    x[k].T @ q @ x[k] + u[k].T @ r @ u[k]

    Parameters
    ----------
    a: state transition matrix.
    b: input matrix.
    q: state cost matrix (semi-positive definite).
    r: input cost matrix (positive definite).
    gamma: discount factor, optional.

    Returns
    -------
    K: ndarray
        The controller gain so that u[k] = K @ x[k].
    P: ndarray
        The cost term of the quadratic value function.
    """
    if gamma is not None:
        a = a * np.sqrt(gamma)
        r = r * (1 / gamma)

    # Compute value function via Ricatti equation
    cost = scipy.linalg.solve_discrete_are(a, b, q, r)

    # Compute the LQR gain
    gain = np.linalg.solve(b.T @ cost @ b + r, b.T @ cost @ a)

    return -gain, cost
