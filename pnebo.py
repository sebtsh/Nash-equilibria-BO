import numpy as np
import tensorflow as tf
import gpflow as gpf
import matplotlib
import pickle
from core.objectives import get_utilities, noisy_observer
from core.optimization import bo_loop_pne
from core.pure_acquisitions import get_acq_pure
from core.utils import (
    get_agent_dims_bounds,
    get_maxmin_mode,
    maxmin_fn,
)
from metrics.regret import calc_regret_pne
from metrics.plotting import plot_utilities_2d, plot_regret
from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path

matplotlib.use("Agg")
ex = Experiment("NashBO-PNE")
ex.observers.append(FileStorageObserver("./runs"))


@ex.named_config
def rand():
    utility_name = "rand"
    acq_name = "ucb_pne"
    agent_dims = [1, 1]  # this determines num_agents and dims
    ls = np.array([0.5] * sum(agent_dims))
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.001
    num_init_points = 5
    num_iters = 200
    beta = 2.0
    n_samples_outer = 50
    seed = 4
    known_best_val = None
    num_actions_discrete = 32
    inner_max_mode = "sample_n_shrink"
    plot_utils = False


@ex.named_config
def gan():
    utility_name = "gan"
    acq_name = "ucb_pne"
    agent_dims = [2, 3]  # this determines num_agents and dims
    ls = np.array([0.5, 0.5, 2.0, 2.0, 2.0])
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.001
    num_init_points = 5
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
    acq_name = "ucb_pne"
    agent_dims = [3, 2]  # this determines num_agents and dims
    ls = np.array([1.5, 0.5, 1.0, 0.5, 0.5])
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.001
    num_init_points = 5
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
    acq_name,
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
    filename = f"pne-{utility_name}-{acq_name}-seed{seed}"

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

    acq_func, args_dict = get_acq_pure(
        acq_name=acq_name,
        beta=beta,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode=maxmin_mode,
        n_samples_outer=n_samples_outer,
        inner_max_mode=inner_max_mode,
        num_actions=num_actions_discrete,
        agent_dims=agent_dims,
    )

    reported_strategies, sampled_strategies, total_time = bo_loop_pne(
        num_agents=num_agents,
        init_data=init_data,
        observer=observer,
        acquisition=acq_func,
        num_iters=num_iters,
        kernel=kernel,
        noise_variance=noise_variance,
        rng=rng,
        args_dict=args_dict,
        save_path=pickles_save_dir + filename,
    )

    time_per_iter = total_time / num_iters

    print("Computing regret for reported strategies")
    reported_sample_regret, reported_cumu_regret = calc_regret_pne(
        u=u,
        data=reported_strategies,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode=maxmin_mode,
        rng=rng,
        known_best_val=known_best_val,
    )

    regrets_save_dir = base_dir + "regrets/"
    plot_regret(
        regret=reported_sample_regret,
        num_iters=num_iters,
        title="Sample pure Nash regret (reported)",
        save=True,
        save_dir=regrets_save_dir,
        filename=filename + "-reported-sample",
    )
    plot_regret(
        regret=reported_cumu_regret,
        num_iters=num_iters,
        title="Cumulative pure Nash regret (reported)",
        save=True,
        save_dir=regrets_save_dir,
        filename=filename + "-reported-cumu",
    )
    print("Regrets of all reported strategies")
    print(reported_sample_regret)
    print(reported_cumu_regret)

    print("Computing regret for sampled strategies")
    sampled_sample_regret, sampled_cumu_regret = calc_regret_pne(
        u=u,
        data=sampled_strategies,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode=maxmin_mode,
        rng=rng,
        known_best_val=known_best_val,
    )

    regrets_save_dir = base_dir + "regrets/"
    plot_regret(
        regret=sampled_sample_regret,
        num_iters=num_iters,
        title="Sample pure Nash regret (sampled)",
        save=True,
        save_dir=regrets_save_dir,
        filename=filename + "-sampled-sample",
    )
    plot_regret(
        regret=sampled_cumu_regret,
        num_iters=num_iters,
        title="Cumulative pure Nash regret (sampled)",
        save=True,
        save_dir=regrets_save_dir,
        filename=filename + "-sampled-cumu",
    )
    print("Regrets of all sampled strategies")
    print(sampled_sample_regret)
    print(sampled_cumu_regret)

    if dims == 2 and plot_utils:
        known_best_point, _ = _, best_val = maxmin_fn(
            outer_funcs=u,
            inner_funcs=u,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=maxmin_mode,
            rng=rng,
            n_samples_outer=100,
        )
        utils_save_dir = base_dir + "utils/"
        plot_utilities_2d(
            u=u,
            bounds=bounds,
            title="Utilities",
            cmap="Spectral",
            save=True,
            save_dir=utils_save_dir,
            filename="utilities",
            show_plot=False,
            known_best_point=known_best_point[None, :],
        )

    pickle.dump(
        (
            reported_strategies,
            sampled_strategies,
            reported_sample_regret,
            reported_cumu_regret,
            sampled_sample_regret,
            sampled_cumu_regret,
            time_per_iter,
            args,
        ),
        open(pickles_save_dir + f"{filename}.p", "wb"),
    )

    print(f"Completed run {run_id} with parameters {args}")
