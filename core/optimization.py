from tqdm import trange
from core.utils import merge_data
from core.models import create_models


def bo_loop(init_data,
            observer,
            models,
            acquisition,
            num_bo_iters,
            kernel,
            noise_variance):
    """
    Main Bayesian optimization loop.
    :param init_data: Tuple (X, Y), X and Y are arrays of shape (n, N).
    :param observer: Callable that takes in an array of shape (n, N) and returns an array of shape (n, N).
    :param models: List of N GPflow GPs.
    :param acquisition: Acquisition function that decides which point to query next.
    :param num_bo_iters: int.
    :param kernel: GPflow kernel.
    :param noise_variance: float.
    :return: Final dataset, tuple (X, Y).
    """
    data = init_data

    for _ in trange(num_bo_iters):
        x_new = acquisition(models)  # (1, N)
        y_new = observer(x_new)
        data = merge_data(data, (x_new, y_new))
        models = create_models(data=data,
                               kernel=kernel,
                               noise_variance=noise_variance)

    return data
