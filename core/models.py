import numpy as np
import gpflow as gpf


def slice_agent_data(data, i):
    """
    Takes all data and returns only the data relevant to the i-th agent's model.
    :param data: Tuple (X, Y), X and Y are arrays of shape (n, N).
    :param i: Agent to slice data for.
    :return: Tuple (X, y_i), y_i is an array of shape (n, 1).
    """
    X, Y = data
    return X, Y[:, i : i + 1]


def create_models(num_agents, data, kernel, noise_variance):
    """
    Creates list of GPs with given data.
    :param num_agents: int.
    :param data: Tuple (X, Y), X and Y are arrays of shape (n, N).
    :param kernel: GPflow kernel.
    :param noise_variance: float.
    :return: List of N GPflow GPs.
    """
    return [
        gpf.models.GPR(
            data=slice_agent_data(data, i), kernel=kernel, noise_variance=noise_variance
        )
        for i in range(num_agents)
    ]


def create_ci_funcs(models, beta):
    """
    Converts GP models into UCB functions and LCB functions.
    :param models: List of N GPflow GPs.
    :param beta: float.
    :return: Tuple, 2 lists of Callables that take in an array of shape (n, dims) and return an array of shape (n, 1).
    """
    N = len(models)

    def create_ci_func(model, is_ucb):
        def inn(X):
            mean, var = model.posterior().predict_f(X)
            if is_ucb:
                return mean + beta * np.sqrt(var)
            else:  # is lcb
                return mean - beta * np.sqrt(var)

        return inn

    return (
        [create_ci_func(models[i], is_ucb=True) for i in range(N)],
        [create_ci_func(models[i], is_ucb=False) for i in range(N)],
    )


def create_mean_funcs(models):
    """
    Converts GP models into their posterior mean functions.
    :param models: List of N GPflow GPs.
    :return: List of Callables that take in an array of shape (n, dims) and return an array of shape (n, 1).
    """
    N = len(models)

    def create_mean_func(model):
        def inn(X):
            mean, _ = model.posterior().predict_f(X)
            return mean

        return inn

    return [create_mean_func(models[i]) for i in range(N)]
