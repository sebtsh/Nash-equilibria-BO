import numpy as np
from tqdm import trange
from core.utils import merge_data
from core.models import create_models
from metrics.plotting import plot_models_2d


def bo_loop_pne_discrete(
    init_data,
    observer,
    models,
    acquisition,
    num_iters,
    kernel,
    noise_variance,
    actions,
    domain,
    plot=False,
    save_dir="",
):
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
        models = create_models(data=data, kernel=kernel, noise_variance=noise_variance)
        if plot:
            plot_models_2d(
                models=models,
                xlims=(0, 1),
                ylims=(0, 1),
                actions=actions,
                domain=domain,
                X=data[0][t : t + 1],
                title=f"GPs iter {t}",
                cmap="Spectral",
                save=True,
                save_dir=save_dir,
                filename=f"gps_{t}",
                show_plot=False,
            )

    return data


def bo_loop_pne(
    init_data, observer, acquisition, num_iters, kernel, noise_variance, rng
):
    data = init_data
    sample_buffer = np.zeros((0, 0))
    for _ in trange(num_iters):
        if len(sample_buffer) == 0:
            models = create_models(
                data=data, kernel=kernel, noise_variance=noise_variance
            )
            sample_buffer = acquisition(models=models, rng=rng)  # (n, N)
        x_new = sample_buffer[0][None, :]
        sample_buffer = np.delete(sample_buffer, 0, axis=0)
        y_new = observer(x_new)
        data = merge_data(data, (x_new, y_new))

    return data


def bo_loop_mne(
    init_data,
    observer,
    acquisition,
    num_iters,
    kernel,
    noise_variance,
    rng,
    plot=False,
    save_dir="",
):
    """
    Main Bayesian optimization loop.
    :param init_data: Tuple (X, Y), X and Y are arrays of shape (n, N).
    :param observer: Callable that takes in an array of shape (n, N) and returns an array of shape (n, N).
    :param acquisition: Acquisition function that decides which point to query next.
    :param num_iters: int.
    :param kernel: GPflow kernel.
    :param noise_variance: float.
    :param actions:
    :param domain:
    :param rng:
    :param plot: bool.
    :param save_dir: str.
    :return: Final dataset, tuple (X, Y).
    """
    data = init_data
    chosen_strategies = []
    sample_buffer = np.zeros((0, 0))
    strategy_buffer = None
    prev_successes = []
    for t in trange(num_iters):
        # print(f"prev_successes: {prev_successes}")
        if len(sample_buffer) == 0:
            models = create_models(
                data=data, kernel=kernel, noise_variance=noise_variance
            )
            sample_buffer, strategy_buffer, prev_successes = acquisition(
                models, prev_successes, rng
            )  # (n, N)
        x_new = sample_buffer[0][None, :]
        sample_buffer = np.delete(sample_buffer, 0, axis=0)
        y_new = observer(x_new)
        data = merge_data(data, (x_new, y_new))
        chosen_strategies.append(strategy_buffer)

        # if plot:
        #     plot_models_2d(
        #         models=models,
        #         xlims=(0, 1),
        #         ylims=(0, 1),
        #         actions=actions,
        #         domain=domain,
        #         X=data[0][t : t + 1],
        #         title=f"GPs iter {t}",
        #         cmap="Spectral",
        #         save=True,
        #         save_dir=save_dir,
        #         filename=f"gps_{t}",
        #         show_plot=False,
        #     )

    return data, chosen_strategies
