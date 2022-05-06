import numpy as np
from core.utils import sigmoid, cross_product


def get_utilities(utility_name, num_agents, bounds, rng, kernel=None, gan_sigma=None):
    """
    Get utility function of each agent.
    :param utility_name:
    :param num_agents:
    :param bounds:
    :param rng:
    :param kernel:
    :param gan_sigma:
    :return: List of Callables that take in an array of shape (n, dims) and return an array of shape (n, 1).
    """
    if utility_name == "randfunc":
        if kernel is None:
            raise Exception("kernel cannot be None for randfunc utility")
        return sample_GP_prior_utilities(
            num_agents=num_agents, kernel=kernel, bounds=bounds, num_points=100, rng=rng
        )
    elif utility_name == "gan":
        return standardize_utilities(u=gan_utilities(rng=rng, gan_sigma=gan_sigma),
                                     bounds=bounds)
    elif utility_name == "bcad":
        raise NotImplementedError
    else:
        raise Exception("Incorrect utility_name passed to get_utilities")


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


def gan_utilities(rng, gan_sigma, m=100):
    """
    WARNING: Only works with 2 agents.
    :param bounds:
    :param rng:
    :return: List of Callables that take in an array of shape (n, 5) and return an array of shape (n, 1).
    """
    # True parameters
    w_real = rng.uniform(low=-1.0, high=1.0, size=(2,))
    print(f"GAN true parameters: {w_real}")
    # Random draws from normal distribution
    z = rng.normal(scale=gan_sigma, size=(m,))

    def logits(w_dis, x):
        """

        :param w_dis: array of size (n, 3).
        :param x: array of size (n, m, 2).
        :return: array of size (n, m). Probabilities of each data point that each discriminator predicts.
        """
        return sigmoid(
            w_dis[:, 0:1] * (x[:, :, 0] ** 2)
            + w_dis[:, 1:2] * (x[:, :, 0] * x[:, :, 1])
            + w_dis[:, 2:3] * (x[:, :, 1] ** 2)
        )

    def utility_gen(w):
        """
        :w: array of size (n, 5).
        :return: array of size (n, 1).
        """
        w_gen = w[:, :2]
        w_dis = w[:, 2:]
        x_gen = w_gen[:, None, :] * z[:, None]  # (n, m, 2)
        return np.expand_dims(
            np.mean(
                np.log(logits(w_dis, x_gen)),
                axis=1,
            ),
            axis=-1,
        )

    def utility_dis(w):
        """
        :param w: array of size (n, 5).
        :return: array of size (n, 1).
        """
        w_gen = w[:, :2]
        w_dis = w[:, 2:]
        x_real = w_real[None, None, :] * z[:, None]  # (1, m, 2)
        x_gen = w_gen[:, None, :] * z[:, None]  # (n, m, 2)

        real_score = np.mean(
            np.log(logits(w_dis, x_real)),
            axis=1,
        )  # (1,)

        fake_score = np.mean(
            np.log(1 - logits(w_dis, x_gen)),
            axis=1,
        )  # (n,)

        return np.expand_dims(real_score + fake_score, axis=-1)

    return [utility_gen, utility_dis]


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


def standardize_utilities(u, bounds, num_samples=10000):
    """
    Produces new utility functions that are approximately standardized for easier learning with GPs.
    :param u: List of Callables that take in an array of shape (n, dims) and return an array of shape (n, 1).
    :param bounds:
    :param num_samples: int.
    :return: List of Callables that take in an array of shape (n, dims) and return an array of shape (n, 1).
    """
    dims = len(bounds)
    num_discrete_each_dim = int(np.ceil(num_samples ** (1 / dims)))
    samples = np.linspace(bounds[0, 0], bounds[0, 1], num_discrete_each_dim)[:, None]
    for i in range(1, dims):
        samples = cross_product(
            samples,
            np.linspace(bounds[i, 0], bounds[i, 1], num_discrete_each_dim)[:, None],
        )
    means = []
    stds = []
    for i in range(len(u)):
        func_vals = u[i](samples)
        means.append(np.mean(func_vals))
        stds.append(np.std(func_vals))
    standardized_utils = []
    for i in range(len(u)):
        standardized_utils.append(lambda x, copy=i: (u[copy](x) - means[copy]) / stds[copy])

    return standardized_utils
