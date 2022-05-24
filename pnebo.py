import numpy as np
import tensorflow as tf
import gpflow as gpf
import matplotlib
import pickle
from core.objectives import get_utilities, noisy_observer
from core.optimization import bo_loop_pne
from core.acquisitions import get_acq_pure
from core.utils import (
    get_agent_dims_bounds,
    discretize_domain,
    cross_product,
    create_response_dict,
    get_maxmin_mode,
)
from metrics.regret import calc_regret_pne, calc_imm_regret_pne
from metrics.plotting import plot_utilities_2d, plot_regret, plot_imm_regret
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
    num_iters = 800
    beta = 2.0
    n_samples_outer = 10
    seed = 0
    known_best_val = None
    num_actions_discrete = 16
    immreg_skip_length = 20


@ex.named_config
def gan():
    utility_name = "gan"
    acq_name = "ucb_pne"
    agent_dims = [2, 3]  # this determines num_agents and dims
    ls = np.array([0.5, 0.5, 2.0, 2.0, 2.0])
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.001
    num_init_points = 5
    num_iters = 1200
    beta = 2.0
    n_samples_outer = 11
    seed = 0
    known_best_val = 0.0
    num_actions_discrete = 32
    immreg_skip_length = 20


@ex.named_config
def bcad():
    utility_name = "bcad"
    acq_name = "ucb_pne"
    agent_dims = [4, 2]  # this determines num_agents and dims
    ls = np.array([1.5, 0.5, 1.5, 0.5, 0.5, 0.5])
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.001
    num_init_points = 5
    num_iters = 1600
    beta = 2.0
    n_samples_outer = 12
    seed = 0
    known_best_val = 0.0
    num_actions_discrete = 32
    immreg_skip_length = 20


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
    immreg_skip_length,
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
    dir = "results/pne/" + utility_name + "/"
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

    if (
        acq_name == "prob_eq" or acq_name == "SUR"
    ):  # Do these steps for discrete acquisitions
        # Discretize the domain
        domain = discretize_domain(
            num_agents=num_agents,
            num_actions=num_actions_discrete,
            bounds=bounds,
            agent_dims=agent_dims,
        )
        # Create response_dicts
        print("Creating response dicts")
        action_idxs = np.arange(num_actions_discrete)
        domain_in_idxs = action_idxs[:, None]
        for i in range(1, num_agents):
            domain_in_idxs = cross_product(domain_in_idxs, action_idxs[:, None])
        response_dicts = [
            create_response_dict(i, domain, domain_in_idxs, action_idxs)
            for i in range(len(agent_dims))
        ]
    else:
        domain = None
        response_dicts = None

    acq_func, args_dict = get_acq_pure(
        acq_name=acq_name,
        beta=beta,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode=maxmin_mode,
        n_samples_outer=n_samples_outer,
        noise_variance=noise_variance,
        domain=domain,
        response_dicts=response_dicts,
        num_actions=num_actions_discrete,
    )

    final_data, total_time = bo_loop_pne(
        num_agents=num_agents,
        init_data=init_data,
        observer=observer,
        acquisition=acq_func,
        num_iters=num_iters,
        kernel=kernel,
        noise_variance=noise_variance,
        rng=rng,
        args_dict=args_dict,
    )

    final_data_minus_init = (
        final_data[0][num_init_points:],
        final_data[1][num_init_points:],
    )
    time_per_iter = total_time / num_iters

    print("Computing regret")
    sample_regret, cumu_regret = calc_regret_pne(
        u=u,
        data=final_data_minus_init,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode=maxmin_mode,
        rng=rng,
        n_samples_outer=n_samples_outer + 5,
        known_best_val=known_best_val,
    )

    regrets_save_dir = dir + "regrets/"
    plot_regret(
        regret=sample_regret,
        num_iters=num_iters,
        title="Sample regret (all samples)",
        save=True,
        save_dir=regrets_save_dir,
        filename=filename + "-sample",
    )
    plot_regret(
        regret=cumu_regret,
        num_iters=num_iters,
        title="Cumulative regret (all samples)",
        save=True,
        save_dir=regrets_save_dir,
        filename=filename + "-cumu",
    )
    print("Regrets of all samples")
    print(sample_regret)
    print(cumu_regret)

    if acq_name == "ucb_pne":
        noreg_seq = np.array(
            [
                j * (num_agents + 1)
                for j in range(int(np.ceil(num_iters / (num_agents + 1))))
            ],
            dtype=np.int32,
        )
        plot_regret(
            regret=sample_regret[noreg_seq],
            num_iters=len(noreg_seq),
            title="Sample regret (no-regret sequence)",
            save=True,
            save_dir=regrets_save_dir,
            filename=filename + "-samplenoreg",
        )
        noreg_cumu_regret = []
        for i in range(len(noreg_seq)):
            noreg_cumu_regret.append(np.sum(sample_regret[noreg_seq][: i + 1]))
        plot_regret(
            regret=noreg_cumu_regret,
            num_iters=len(noreg_seq),
            title="Cumulative regret (no-regret sequence)",
            save=True,
            save_dir=regrets_save_dir,
            filename=filename + "-cumunoreg",
        )
        print("Regrets of no-regret sequence")
        print(sample_regret[noreg_seq])
        print(cumu_regret[noreg_seq])

    if dims == 2:
        utils_save_dir = dir + "utils/"
        plot_utilities_2d(
            u=u,
            bounds=bounds,
            title="Utilities",
            cmap="Spectral",
            save=True,
            save_dir=utils_save_dir,
            filename="utilities",
            show_plot=False,
        )

    pickles_save_dir = dir + "pickles/"
    Path(pickles_save_dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(
        (final_data, sample_regret, cumu_regret, time_per_iter, args),
        open(pickles_save_dir + f"{filename}.p", "wb"),
    )

    print("Calculating immediate regret")
    imm_regret = calc_imm_regret_pne(
        u=u,
        data=final_data,
        num_agents=num_agents,
        num_init_points=num_init_points,
        kernel=kernel,
        noise_variance=noise_variance,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode=maxmin_mode,
        rng=rng,
        n_samples_outer=n_samples_outer,
        known_best_val=known_best_val,
        skip_length=immreg_skip_length,
    )
    plot_imm_regret(
        regret=imm_regret,
        num_iters=num_iters,
        skip_length=immreg_skip_length,
        title="Immediate regret",
        save=True,
        save_dir=regrets_save_dir,
        filename=filename + "-immreg",
    )

    pickle.dump(
        (final_data, sample_regret, cumu_regret, time_per_iter, args, imm_regret),
        open(pickles_save_dir + f"{filename}-2.p", "wb"),
    )

    print(f"Completed run {run_id} with parameters {args}")
