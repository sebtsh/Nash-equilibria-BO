import gpflow as gpf


def slice_agent_data(data,
                     i):
    """
    Takes all data and returns only the data relevant to the i-th agent's model.
    :param data: Tuple (X, Y), X and Y are arrays of shape (n, N).
    :param i: Agent to slice data for.
    :return: Tuple (X, y_i), y_i is an array of shape (n, 1).
    """
    X, Y = data
    return X, Y[:, i:i+1]


def create_models(data,
                  kernel,
                  noise_variance):
    """
    Creates list of GPs with given data.
    :param data: Tuple (X, Y), X and Y are arrays of shape (n, N).
    :param kernel: GPflow kernel.
    :param noise_variance: float.
    :return: List of N GPflow GPs.
    """
    num_agents = data[0].shape[-1]
    return [gpf.models.GPR(data=slice_agent_data(data, i),
                           kernel=kernel,
                           noise_variance=noise_variance) for i in range(num_agents)]
