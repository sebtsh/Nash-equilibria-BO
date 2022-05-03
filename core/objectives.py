import numpy as np


def sample_GP_prior(kernel, bounds, num_points, rng, jitter=1e-06):
    """
    Sample a random function from a GP prior with mean 0 and covariance specified by a kernel.
    :param kernel: a GPflow kernel.
    :param bounds: array of shape (dims, 2).
    :param num_points: int.
    :param rng: NumPy rng object.
    :param jitter: float.
    :return: Callable that takes in an array of shape (n, N) and returns an array of shape (n, 1).
    """
    n = len(bounds)
    lowers = bounds[:, 0]
    uppers = bounds[:, 1]
    points = rng.uniform(low=lowers, high=uppers, size=(num_points, n))
    cov = kernel(points) + jitter * np.eye(len(points))
    f_vals = rng.multivariate_normal(np.zeros(num_points), cov + jitter)[:, None]
    L_inv = np.linalg.inv(np.linalg.cholesky(cov))
    K_inv_f = L_inv.T @ L_inv @ f_vals
    return lambda x: kernel(x, points) @ K_inv_f


def sample_GP_prior_utilities(num_agents, kernel, bounds, num_points, rng):
    """
    Creates agent utilities represented as a list of functions.
    :param num_agents: int.
    :param kernel: a GPflow kernel.
    :param bounds: array of shape (dims, 2).
    :param num_points: int.
    :param rng: NumPy rng object.
    :return: List of Callables that take in an array of shape (n, dims) and return an array of shape (n, 1).
    """
    return [sample_GP_prior(kernel, bounds, num_points, rng) for _ in range(num_agents)]


def noisy_observer(u, noise_variance, rng):
    """
    Creates a list of functions that take in a strategy and return the noisy utility value for that agent.
    :param u: List of N Callables that each take in an array of shape (n, dims) and return an array of shape (n, 1).
    :param noise_variance: float.
    :param rng: NumPy rng object.
    :return: Callable that takes in an array of shape (n, dims) and returns an array of shape (n, N).
    """
    N = len(u)

    def observer(X):
        n, _ = X.shape
        Y = np.zeros((n, N))
        for i in range(N):
            Y[:, i] = np.squeeze(
                u[i](X) + rng.normal(0, noise_variance, (n, 1)), axis=-1
            )
        return Y

    return observer
