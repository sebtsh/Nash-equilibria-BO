import numpy as np
from core.utils import sigmoid, cross_product


def get_utilities(utility_name, num_agents, bounds, rng, kernel=None, gan_sigma=1.0):
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
    if utility_name == "rand":
        if kernel is None:
            raise Exception("kernel cannot be None for rand utility")
        return (
            sample_GP_prior_utilities(
                num_agents=num_agents,
                kernel=kernel,
                bounds=bounds,
                num_points=100,
                rng=rng,
            ),
            None,
        )
    elif utility_name == "gan":
        u, info = gan_utilities(rng=rng, gan_sigma=gan_sigma)
        return standardize_utilities(u=u, bounds=bounds, std=False), info
    elif utility_name == "bcad":
        return bcad_utilities(rect_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]), rng=rng)
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

    return [utility_gen, utility_dis], (w_real, z)


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


def standardize_utilities(u, bounds, num_samples=10000, std=True):
    """
    Produces new utility functions that are approximately standardized for easier learning with GPs.
    :param u: List of Callables that take in an array of shape (n, dims) and return an array of shape (n, 1).
    :param bounds:
    :param num_samples: int.
    :param std: bool. Divide by standard deviation or not.
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
        if std:
            standardized_utils.append(
                lambda x, copy=i: (u[copy](x) - means[copy]) / stds[copy]
            )
        else:
            standardized_utils.append(lambda x, copy=i: u[copy](x) - means[copy])

    return standardized_utils


def sample_rectangle_2d(bounds, rng):
    """
    Samples a 2d rectangle of area 1 within bounds.
    :param bounds: array of shape (dims, 2).
    :param rng: NumPy rng object.
    :return: tuple of (low_x1, high_x1, low_x2, high_x2).
    """
    min_x1, max_x1 = bounds[0]
    min_x2, max_x2 = bounds[1]
    max_length, max_height = bounds[:, 1] - bounds[:, 0]

    # Sample length (x1)
    low_x1 = rng.uniform(low=min_x1, high=max_x1 - 1 / max_height)
    high_x1 = rng.uniform(low=low_x1 + 1 / max_height, high=max_x1)
    x1_length = high_x1 - low_x1

    # Sample height (x2)
    x2_height = 1 / x1_length
    low_x2 = rng.uniform(low=min_x2, high=max_x1 - x2_height)
    high_x2 = low_x2 + x2_height

    return low_x1, high_x1, low_x2, high_x2


def bcad_utilities(rect_bounds, rng, m=20):
    """
    WARNING: Only works with two agents.
    :param rect_bounds: array of shape (2, 2).
    :param rng: NumPy rng object.
    :param m: int.
    :return: List of 2 Callables that each take in an array of shape (n, 6) and return an array of shape (n, 1).
    """
    margin = 1.0
    rect = sample_rectangle_2d(rect_bounds, rng)
    low_x1, high_x1, low_x2, high_x2 = rect

    X_s = cross_product(
        np.linspace(low_x1, high_x1, m)[:, None],
        np.linspace(low_x2, high_x2, m)[:, None],
    )

    def g_func(X):
        return (
            0.25 * (X[..., 0:1] ** 2)
            - 0.5 * (X[..., 0:1] * X[..., 1:2])
            - 0.25 * (X[..., 1:2] ** 2)
        )  # (..., 1)

    def grad_g_func(X):
        return 0.5 * np.concatenate(
            [X[..., 0:1] - X[..., 1:2], -X[..., 0:1] - X[..., 1:2]], axis=-1
        )  # (..., 2)

    def g_linear_func(X, perturbed_X):
        return (
            g_func(X) + np.sum(grad_g_func(X) * (perturbed_X - X), axis=-1)[..., None]
        )  # (..., 1)

    def utility_attacker(params):
        """
        Deterministic estimate of attacker's utility. Negative of the defender's utility.
        :param params: array of shape (n, 6).
        :return: array of shape (n, 1).
        """
        v = params[:, :4].copy()  # (n, 4)
        d = params[:, 4:].copy()  # (n, 2)

        # Attacker's perturbations
        signs = np.sign(g_func(X_s))  # (m ** 2, 1)
        b_vals = np.concatenate(
            [
                v[:, 0:1, None] * X_s[:, 0:1] + v[:, 1:2, None] * X_s[:, 1:2],
                v[:, 2:3, None] * X_s[:, 0:1] + v[:, 3:4, None] * X_s[:, 1:2],
            ],
            axis=-1,
        )  # (n, m ** 2, 2)
        b_vals_norm = np.linalg.norm(b_vals, axis=-1)[..., None]  # (n, m ** 2, 1)
        # Some norms might be zero, so we change their norm to 1 to avoid division by zero. Perturbation will be zero
        zero_indices = np.array(
            list(set(np.where(np.squeeze(b_vals_norm, axis=-1) == 0.0)[0]))
        )
        if len(zero_indices) != 0:
            b_vals_norm[zero_indices] = 1.0

        attacker_perturbs = -margin * signs * b_vals / b_vals_norm  # (n, m ** 2, 2)

        # Defender's perturbations
        defender_perturbs = np.expand_dims(d, axis=1)  # (n, 1, 2)
        # Ensure perturbations are not larger than margin
        norms = np.linalg.norm(defender_perturbs, axis=-1)  # (n, 1)
        larger_idxs = np.where(norms > margin)[0]
        defender_perturbs[larger_idxs] = (
            defender_perturbs[larger_idxs] / norms[larger_idxs][..., None] * margin
        )

        perturbed_X = X_s + attacker_perturbs + defender_perturbs  # (n, m ** 2, 2)
        X_expanded = np.tile(X_s, (len(v), 1, 1))
        g_L_vals = g_linear_func(X_expanded, perturbed_X)  # (n, m ** 2, 1)
        g_vals = g_func(X_expanded)  # (n, m ** 2, 1)

        func_signs = np.sign(g_vals)
        perturbed_signs = np.sign(g_L_vals)

        return np.mean(func_signs * perturbed_signs * -1, axis=1)  # (n, 1)

    return [utility_attacker, lambda x: -utility_attacker(x)], (rect, X_s)
