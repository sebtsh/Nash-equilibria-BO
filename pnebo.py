import numpy as np
import tensorflow as tf
import gpflow as gpf
import matplotlib
from core.objectives import get_utilities, noisy_observer
from core.optimization import bo_loop_pne
from core.acquisitions import get_acquisition
from core.utils import get_agent_dims_bounds
from metrics.regret import calc_regret_pne
from metrics.plotting import plot_utilities_2d, plot_regret
from sacred import Experiment
from sacred.observers import FileStorageObserver

matplotlib.use("Agg")
ex = Experiment("NashBO")
ex.observers.append(FileStorageObserver("../runs"))


@ex.named_config
def randfunc():
    utility_name = "randfunc"
    acq_name = "ucb_pne"  # 'ucb_pne_naive', 'ucb_pne'
    agent_dims = [1, 1]  # this determines num_agents and dims
    lengthscale = 0.5
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.01
    num_init_points = 5
    num_iters = 500
    beta = 2.0
    maxmin_mode = "DIRECT"
    n_samples_outer = 10
    seed = 0
    known_best_val = None


@ex.named_config
def gan():
    utility_name = "gan"
    acq_name = "ucb_pne"  # 'ucb_pne_naive', 'ucb_pne'
    agent_dims = [2, 3]  # this determines num_agents and dims
    lengthscale = 0.5
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    noise_variance = 0.01
    num_init_points = 5
    num_iters = 1000
    beta = 2.0
    maxmin_mode = "random"
    n_samples_outer = 12
    seed = 0
    known_best_val = 0.0


@ex.automain
def main(
    utility_name,
    acq_name,
    agent_dims,
    lengthscale,
    bound,
    noise_variance,
    num_init_points,
    num_iters,
    beta,
    maxmin_mode,
    n_samples_outer,
    seed,
    known_best_val,
):
    args = dict(sorted(locals().items()))
    print(f"Running with parameters {args}")
    run_id = ex.current_run._id

    num_agents = len(agent_dims)
    dims = np.sum(agent_dims)
    bounds = np.array([bound for _ in range(dims)])
    ls = np.array([lengthscale] * dims)
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    dir = "runs/" + utility_name + "/"

    kernel = gpf.kernels.SquaredExponential(lengthscales=ls)
    gan_sigma = 1.0
    u = get_utilities(
        utility_name=utility_name,
        num_agents=num_agents,
        bounds=bounds,
        rng=rng,
        kernel=kernel,
        gan_sigma=gan_sigma,
    )

    observer = noisy_observer(u=u, noise_variance=noise_variance, rng=rng)
    init_X = rng.uniform(
        low=bounds[:, 0], high=bounds[:, 1], size=(num_init_points, dims)
    )
    init_data = (init_X, observer(init_X))

    agent_dims_bounds = get_agent_dims_bounds(agent_dims=agent_dims)
    acq_func = get_acquisition(
        acq_name=acq_name,
        beta=beta,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode=maxmin_mode,
        n_samples_outer=n_samples_outer,
    )

    final_data = bo_loop_pne(
        init_data=init_data,
        observer=observer,
        acquisition=acq_func,
        num_iters=num_iters,
        kernel=kernel,
        noise_variance=noise_variance,
        rng=rng,
    )

    final_data_minus_init = (
        final_data[0][num_init_points:],
        final_data[1][num_init_points:],
    )

    print("Computing regret")
    imm_regret, cumu_regret = calc_regret_pne(
        u=u,
        data=final_data_minus_init,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode=maxmin_mode,
        rng=rng,
        n_samples_outer=n_samples_outer,
        known_best_val=known_best_val,
    )

    regrets_save_dir = dir + "regrets/"

    plot_regret(
        regret=imm_regret,
        num_iters=num_iters,
        title="Immediate regret (all samples)",
        save=True,
        save_dir=regrets_save_dir,
        filename="imm-all",
    )
    plot_regret(
        regret=cumu_regret,
        num_iters=num_iters,
        title="Cumulative regret (all samples)",
        save=True,
        save_dir=regrets_save_dir,
        filename="cumu-all",
    )

    noreg_seq = np.array(
        [
            j * (num_agents + 1)
            for j in range(int(np.ceil(num_iters / (num_agents + 1))))
        ],
        dtype=np.int32,
    )
    plot_regret(
        regret=imm_regret[noreg_seq],
        num_iters=len(noreg_seq),
        title="Immediate regret (no-regret sequence)",
        save=True,
        save_dir=regrets_save_dir,
        filename="imm-noregseq",
    )
    plot_regret(
        regret=cumu_regret[noreg_seq],
        num_iters=len(noreg_seq),
        title="Cumulative regret (no-regret sequence)",
        save=True,
        save_dir=regrets_save_dir,
        filename="cumu-noregseq",
    )

    print("Regrets of all samples")
    print(imm_regret)
    print(cumu_regret)
    print("Regrets of no-regret sequence")
    print(imm_regret[noreg_seq])
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

    print(f"Completed run {run_id} with parameters {args}")
