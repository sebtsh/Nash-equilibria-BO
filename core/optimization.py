from tqdm import trange
from core.utils import merge_data
from core.models import create_models
from metrics.plotting import plot_models_2d


def bo_loop(init_data,
            observer,
            models,
            acquisition,
            num_iters,
            kernel,
            noise_variance,
            actions,
            domain,
            plot=False,
            save_dir=""):
    """
    Main Bayesian optimization loop.
    :param init_data: Tuple (X, Y), X and Y are arrays of shape (n, N).
    :param observer: Callable that takes in an array of shape (n, N) and returns an array of shape (n, N).
    :param models: List of N GPflow GPs.
    :param acquisition: Acquisition function that decides which point to query next.
    :param num_iters: int.
    :param kernel: GPflow kernel.
    :param noise_variance: float.
    :param actions:
    :param domain:
    :param plot: bool.
    :param save_dir: str.
    :return: Final dataset, tuple (X, Y).
    """
    data = init_data

    for t in trange(num_iters):
        X_new = acquisition(models)  # (n, N)
        y_new = observer(X_new)
        data = merge_data(data, (X_new, y_new))
        models = create_models(data=data,
                               kernel=kernel,
                               noise_variance=noise_variance)
        if plot:
            plot_models_2d(models=models,
                           xlims=(0, 1),
                           ylims=(0, 1),
                           actions=actions,
                           domain=domain,
                           X=data[0][t:t+1],
                           title=f"GPs iter {t}",
                           cmap="Spectral",
                           save=True,
                           save_dir=save_dir,
                           filename=f"gps_{t}",
                           show_plot=False)

    return data
