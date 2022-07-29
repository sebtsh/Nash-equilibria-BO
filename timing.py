import numpy as np
import tensorflow as tf
import gpflow as gpf
import matplotlib
import pickle
from core.objectives import get_utilities, noisy_observer
from core.pure_acquisitions import get_acq_pure
from core.utils import (
    get_agent_dims_bounds,
    get_maxmin_mode,
)
from core.models import create_models

from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path
from time import process_time

matplotlib.use("Agg")
ex = Experiment("NashBO-PNE")
ex.observers.append(FileStorageObserver("./runs"))


@ex.named_config
def rand():
    utility_name = "rand"
    agent_dims = [1, 1]  # this determines num_agents and dims
    ls = np.array([0.5] * sum(agent_dims))
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.001
    num_init_points = 100
    num_iters = 200
    beta = 2.0
    n_samples_outer = 50
    seed = 0
    known_best_val = None
    num_actions_discrete = 16
    inner_max_mode = "sample_n_shrink"
    plot_utils = False


@ex.named_config
def gan():
    utility_name = "gan"
    agent_dims = [2, 3]  # this determines num_agents and dims
    ls = np.array([0.5, 0.5, 2.0, 2.0, 2.0])
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.001
    num_init_points = 100
    num_iters = 600
    beta = 2.0
    n_samples_outer = 50
    seed = 0
    known_best_val = 0.0
    num_actions_discrete = 32
    inner_max_mode = "sample_n_shrink"
    plot_utils = False


@ex.named_config
def bcad():
    utility_name = "bcad"
    agent_dims = [3, 2]  # this determines num_agents and dims
    ls = np.array([1.5, 0.5, 1.0, 0.5, 0.5])
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.001
    num_init_points = 100
    num_iters = 600
    beta = 2.0
    n_samples_outer = 50
    seed = 0
    known_best_val = 0.0
    num_actions_discrete = 32
    inner_max_mode = "sample_n_shrink"
    plot_utils = False


@ex.automain
def main(
    utility_name,
    agent_dims,
    ls,
    bound,
    noise_variance,
    num_init_points,
    num_iters,
    beta,
    n_samples_outer,
    seed,
    known_best_val,
    num_actions_discrete,
    inner_max_mode,
    plot_utils,
):
    args = dict(sorted(locals().items()))
    print(f"Running with parameters {args}")
    run_id = ex.current_run._id

    maxmin_mode = get_maxmin_mode()
    num_agents = len(agent_dims)
    dims = np.sum(agent_dims)
    bounds = np.array([bound for _ in range(dims)])
    agent_dims_bounds = get_agent_dims_bounds(agent_dims=agent_dims)
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    base_dir = "results/pne/" + utility_name + "/"
    pickles_save_dir = base_dir + "pickles/"
    Path(pickles_save_dir).mkdir(parents=True, exist_ok=True)

    kernel = gpf.kernels.SquaredExponential(lengthscales=ls)
    u, _ = get_utilities(
        utility_name=utility_name,
        num_agents=num_agents,
        bounds=bounds,
        rng=rng,
        kernel=kernel,
    )

    observer = noisy_observer(u=u, noise_variance=noise_variance, rng=rng)
    init_X = rng.uniform(
        low=bounds[:, 0], high=bounds[:, 1], size=(num_init_points, dims)
    )
    init_data = (init_X, observer(init_X))

    time_dict = {}
    for acq in ["ucb_pne", "prob_eq", "prob_eq2", "BN"]:
        print(f"Timing for {acq}")
        if acq == "prob_eq2":
            acq_name = "prob_eq"
            num_discrete = num_actions_discrete * 2
        else:
            acq_name = acq
            num_discrete = num_actions_discrete
        acq_func, args_dict = get_acq_pure(
            acq_name=acq_name,
            beta=beta,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=maxmin_mode,
            n_samples_outer=n_samples_outer,
            inner_max_mode=inner_max_mode,
            num_actions=num_discrete,
            agent_dims=agent_dims,
        )
        args_dict["is_reporting"] = False
        models = create_models(
            num_agents=num_agents,
            data=init_data,
            kernel=kernel,
            noise_variance=noise_variance,
        )
        times = []

        for i in range(5):
            start = process_time()
            reported_strategy, sampled_strategy, args_dict = acq_func(
                models=models, rng=rng, args_dict=args_dict
            )
            end = process_time()
            times.append(end - start)
        print(f"Times for {utility_name}-{acq}: {times}")
        print(f"Mean time for {utility_name}-{acq}: {np.mean(times)}")
        time_dict[acq] = times

    pickle.dump(
        time_dict,
        open(pickles_save_dir + f"{utility_name}-timing.p", "wb"),
    )
