import numpy as np
from tqdm import trange
from core.utils import merge_data
from core.models import create_models
from time import process_time


def bo_loop_pne(
    num_agents,
    init_data,
    observer,
    acquisition,
    num_iters,
    kernel,
    noise_variance,
    rng,
    args_dict,
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
    :param args_dict:
    :return:
    """
    data = init_data
    reported_strategies = []
    sampled_strategies = []
    start = process_time()
    for _ in trange(num_iters):
        models = create_models(
            num_agents=num_agents,
            data=data,
            kernel=kernel,
            noise_variance=noise_variance,
        )
        reported_strategy, sampled_strategy, args_dict = acquisition(
            models=models, rng=rng, args_dict=args_dict
        )  # acquisitions must return 1-D arrays of shape (dims, )
        y_new = observer(sampled_strategy[None, :])
        data = merge_data(data, (sampled_strategy[None, :], y_new))
        reported_strategies.append(reported_strategy)
        sampled_strategies.append(sampled_strategy)
    end = process_time()
    total_time = end - start
    reported_strategies = np.array(reported_strategies)
    sampled_strategies = np.array(sampled_strategies)
    return reported_strategies, sampled_strategies, total_time


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
    reported_strategies = []
    sampled_strategies = []
    prev_successes = []
    start = process_time()
    for _ in trange(num_iters):
        # print(f"prev_successes: {prev_successes}")
        models = create_models(
            num_agents=num_agents,
            data=data,
            kernel=kernel,
            noise_variance=noise_variance,
        )
        reported_mixed_strategy, sampled_pure_strategy, prev_successes = acquisition(
            models, prev_successes, rng
        )  # acquisitions need to return (s1, s2) and an array of shape (dims,)
        y_new = observer(sampled_pure_strategy[None, :])
        data = merge_data(data, (sampled_pure_strategy[None, :], y_new))
        reported_strategies.append(reported_mixed_strategy)
        sampled_strategies.append(sampled_pure_strategy)
    end = process_time()
    total_time = end - start
    reported_strategies = np.array(reported_strategies)
    sampled_strategies = np.array(sampled_strategies)
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

    return reported_strategies, sampled_strategies, total_time
