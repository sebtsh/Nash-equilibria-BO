import numpy as np
import tensorflow as tf
import gpflow as gpf
import pickle
import matplotlib
from core.objectives import get_utilities, noisy_observer
from core.utils import cross_product, sobol_sequence, get_agent_dims_bounds
from core.optimization import bo_loop_mne
from core.acquisitions import get_acq_mixed
from core.mne import SEM, neg_brp_mixed
from metrics.regret import calc_regret_mne
from metrics.plotting import plot_regret
from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path

matplotlib.use("Agg")
ex = Experiment("NashBO-MNE")
ex.observers.append(FileStorageObserver("./runs"))


@ex.named_config
def rand():
    utility_name = "rand"
    acq_name = "ucb_mne"
    agent_dims = [1, 1]  # this determines num_agents and dims
    ls = np.array([0.5] * sum(agent_dims))
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    num_actions = 32
    noise_variance = 0.001
    num_init_points = 5
    num_iters = 800
    beta = 2.0
    seed = 0


@ex.named_config
def gan():
    utility_name = "gan"
    acq_name = "ucb_mne"
    agent_dims = [2, 3]  # this determines num_agents and dims
    ls = np.array([0.5, 0.5, 2.0, 2.0, 2.0])
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    num_actions = 64
    noise_variance = 0.001
    num_init_points = 5
    num_iters = 1200
    beta = 2.0
    seed = 0


@ex.named_config
def bcad():
    utility_name = "bcad"
    acq_name = "ucb_mne"
    agent_dims = [4, 2]  # this determines num_agents and dims
    ls = np.array([1.5, 0.5, 1.5, 0.5, 0.5, 0.5])
    bound = [-1.0, 1.0]  # assumes same bounds for all dims
    num_actions = 128
    noise_variance = 0.001
    num_init_points = 5
    num_iters = 1600
    beta = 2.0
    seed = 0


@ex.automain
def main(
    utility_name,
    acq_name,
    agent_dims,
    ls,
    bound,
    num_actions,
    noise_variance,
    num_init_points,
    num_iters,
    beta,
    seed,
):
    args = dict(sorted(locals().items()))
    print(f"Running with parameters {args}")
    run_id = ex.current_run._id

    num_agents = len(agent_dims)
    dims = np.sum(agent_dims)
    bounds = np.array([bound for _ in range(dims)])
    agent_dims_bounds = get_agent_dims_bounds(agent_dims=agent_dims)
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    dir = "results/mne/" + utility_name + "/"
    filename = f"mne-{utility_name}-{acq_name}-seed{seed}"

    bounds = np.array([bound for _ in range(dims)])
    agent_dims_bounds = get_agent_dims_bounds(agent_dims=agent_dims)
    start_dim, end_dim = agent_dims_bounds[0]
    domain = sobol_sequence(num_points=num_actions, bounds=bounds[start_dim:end_dim])
    for i in range(1, num_agents):
        start_dim, end_dim = agent_dims_bounds[i]
        domain = cross_product(
            domain,
            sobol_sequence(num_points=num_actions, bounds=bounds[start_dim:end_dim]),
        )

    kernel = gpf.kernels.SquaredExponential(lengthscales=ls)

    u, _ = get_utilities(
        utility_name=utility_name,
        num_agents=num_agents,
        bounds=bounds,
        rng=rng,
        kernel=kernel,
    )
    U1 = np.reshape(u[0](domain), (num_actions, num_actions))
    U2 = np.reshape(u[1](domain), (num_actions, num_actions))
    # print("U1:")
    # print(U1)
    # print("U2:")
    # print(U2)
    #
    # all_res, _ = SEM(U1, U2, "all")
    # for res in all_res:
    #     s1 = res[:num_actions]
    #     s2 = res[num_actions + 1 : num_actions + num_actions + 1]
    #     print(f"Agent 1 strategy: {s1}, with value {res[num_actions]}")
    #     print(f"Agent 2 strategy: {s2}, with value {res[num_actions + num_actions + 1]}")
    #     print(f"-brp: {neg_brp_mixed(U1, U2, (s1, s2))}")
    #     print("==============")

    observer = noisy_observer(u=u, noise_variance=noise_variance, rng=rng)
    init_idxs = rng.integers(
        low=0, high=num_actions**num_agents, size=num_init_points
    )
    init_X = domain[init_idxs]
    init_data = (init_X, observer(init_X))

    acq_func = get_acq_mixed(
        acq_name=acq_name, beta=beta, domain=domain, num_actions=num_actions
    )

    final_data, chosen_strategies = bo_loop_mne(
        init_data=init_data,
        observer=observer,
        acquisition=acq_func,
        num_iters=num_iters,
        kernel=kernel,
        noise_variance=noise_variance,
        rng=rng,
        plot=False,
    )

    imm_regret, cumu_regret = calc_regret_mne(
        strategies=chosen_strategies, U1=U1, U2=U2
    )
    print("Immediate regret:")
    print(imm_regret)
    print("Cumulative regret:")
    print(cumu_regret)
    regrets_save_dir = dir + "regrets/"
    plot_regret(
        regret=imm_regret,
        num_iters=num_iters,
        title=f"MNE-{utility_name}: Immediate regret",
        save=True,
        save_dir=regrets_save_dir,
        filename=filename + "-imm",
    )
    plot_regret(
        regret=cumu_regret,
        num_iters=num_iters,
        title=f"MNE-{utility_name}: Cumulative regret",
        save=True,
        save_dir=regrets_save_dir,
        filename=filename + "-cumu",
    )
    pickles_save_dir = dir + "pickles/"
    Path(pickles_save_dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(
        (final_data, chosen_strategies, imm_regret, cumu_regret),
        open(pickles_save_dir + f"{filename}.p", "wb"),
    )

    print(f"Completed run {run_id} with parameters {args}")
