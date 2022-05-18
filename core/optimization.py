import numpy as np
from tqdm import trange
from core.utils import merge_data
from core.models import create_models


def bo_loop_pne(
    num_agents, init_data, observer, acquisition, num_iters, kernel, noise_variance, rng
):
    """
    Main Bayesian optimization loop for PNEs.
    :param num_agents:
    :param init_data:
    :param observer:
    :param acquisition:
    :param num_iters:
    :param kernel:
    :param noise_variance:
    :param rng:
    :return:
    """
    data = init_data
    sample_buffer = np.zeros((0, 0))
    for _ in trange(num_iters):
        if len(sample_buffer) == 0:
            models = create_models(
                num_agents=num_agents,
                data=data,
                kernel=kernel,
                noise_variance=noise_variance,
            )
            sample_buffer = acquisition(models=models, rng=rng)  # (n, N)
        x_new = sample_buffer[0][None, :]
        sample_buffer = np.delete(sample_buffer, 0, axis=0)
        y_new = observer(x_new)
        data = merge_data(data, (x_new, y_new))

    return data


def bo_loop_mne(
    num_agents,
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
    Main Bayesian optimization loop for MNEs.
    :param num_agents: int.
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
    for _ in trange(num_iters):
        # print(f"prev_successes: {prev_successes}")
        if len(sample_buffer) == 0:
            models = create_models(
                num_agents=num_agents,
                data=data,
                kernel=kernel,
                noise_variance=noise_variance,
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
