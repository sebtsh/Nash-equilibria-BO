import numpy as np


def sample_GP_prior(kernel,
                    lowers,
                    uppers,
                    num_points,
                    rng,
                    jitter=1e-06):
    """
    Sample a random function from a GP prior with mean 0 and covariance specified by a kernel.
    :param kernel: a GPflow kernel.
    :param lowers: array of shape (n, ).
    :param uppers: array of shape (n, ).
    :param num_points: int.
    :param rng: NumPy rng object.
    :param jitter: float.
    :return: Callable that takes in an array of shape (m, d) and returns an array of shape (m, 1).
    """
    n = len(lowers)
    assert n == len(uppers)
    points = rng.uniform(low=lowers, high=uppers, size=(num_points, n))
    cov = kernel(points) + jitter * np.eye(len(points))
    f_vals = rng.multivariate_normal(np.zeros(num_points), cov + jitter)[:, None]
    L_inv = np.linalg.inv(np.linalg.cholesky(cov))
    K_inv_f = L_inv.T @ L_inv @ f_vals
    return lambda x: kernel(x, points) @ K_inv_f


def sample_GP_prior_utilities(num_agents,
                              kernel,
                              lowers,
                              uppers,
                              num_points,
                              rng):
    """
    Creates agent utilities represented as a list of functions.
    :param num_agents: int.
    :param kernel: a GPflow kernel.
    :param lowers: array of shape (n, ).
    :param uppers: array of shape (n, ).
    :param num_points: int.
    :param rng: NumPy rng object.
    :return: List of Callables that take in an array of shape (m, d) and return an array of shape (m, 1).
    """
    return [sample_GP_prior(kernel,
                            lowers,
                            uppers,
                            num_points,
                            rng) for _ in range(num_agents)]
