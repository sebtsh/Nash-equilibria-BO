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
    discretize_domain,
    cross_product,
    create_response_dict,
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
    agent_dims = [1, 1]  # this determines num_agents and dims
    ls = np.array([0.5] * sum(agent_dims))
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.001
    num_init_points = 5
    num_iters = 200
    beta = 2.0
    n_samples_outer = 50
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
    num_init_points = 5
    num_iters = 600
    beta = 2.0
    n_samples_outer = 50
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
    num_init_points = 5
    num_iters = 600
    beta = 2.0
    n_samples_outer = 50
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

    for seed in range(5):
        print(f"Seed {seed}")
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

        base_dir = "results/pne/" + utility_name + "/"
        save_dir = base_dir
        pickles_dir = base_dir + "pickles/"

        if utility_name == "rand":
            _, known_best_val = maxmin_fn(
                outer_funcs=u,
                inner_funcs=u,
                bounds=bounds,
                agent_dims_bounds=agent_dims_bounds,
                mode=maxmin_mode,
                rng=rng,
                n_samples_outer=200,
            )

        for acquisition in ["ucb_pne", "prob_eq", "BN"]:
            print(f"acquisition {acquisition}")
            filename = f"pne-{utility_name}-{acquisition}-seed{seed}.p"
            (
                reported_strategies,
                sampled_strategies,
                reported_sample_regret,
                reported_cumu_regret,
                time_per_iter,
                args,
            ) = pickle.load(open(pickles_dir + filename, "rb"))

            print("Sanity check")
            reported_sample_regret2, reported_cumu_regret2 = calc_regret_pne(
                u=u,
                data=reported_strategies[:5],
                bounds=bounds,
                agent_dims_bounds=agent_dims_bounds,
                mode=maxmin_mode,
                rng=rng,
                known_best_val=known_best_val,
            )

            if np.allclose(reported_sample_regret2, reported_sample_regret[:5]):
                print("Sanity check passed")
            else:
                print("Sanity check failed")
                print(f"Previously computed regrets: {reported_sample_regret[:5]}")
                print(f"Newly computed regrets: {reported_sample_regret2}")

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
                open(pickles_save_dir + filename[:-2] + "-2" + ".p", "wb"),
            )
