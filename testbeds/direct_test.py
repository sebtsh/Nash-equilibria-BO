import numpy as np
import tensorflow as tf
import gpflow as gpf
import time
from core.objectives import sample_GP_prior_utilities, noisy_observer
from core.models import create_models
from core.acquisitions import get_acquisition, create_ci_funcs
from core.utils import get_agent_dims_bounds
from core.pne import evaluate_sample
from metrics.plotting import plot_utilities_2d

# Parameters
acq_name = "ucb_pne"  # 'ucb_pne_naive', 'ucb_pne'
seed = 0
agent_dims = [1, 1]  # this determines num_agents and dims
num_agents = len(agent_dims)
dims = np.sum(agent_dims)
ls = np.array([1.0] * dims)
bounds = np.array([[-1.0, 1.0] for _ in range(dims)])
noise_variance = 0.1
num_init_points = 20
num_iters = 50
beta = 2.0
maxmin_mode = "random"
n_samples_outer = 3
plot_utils = False
dir = "results/testcont/"
rng = np.random.default_rng(seed)
tf.random.set_seed(seed)

kernel = gpf.kernels.SquaredExponential(lengthscales=ls)
u = sample_GP_prior_utilities(
    num_agents=num_agents, kernel=kernel, bounds=bounds, num_points=100, rng=rng
)
if plot_utils:
    utils_save_dir = dir + "gif/"
    plot_utilities_2d(
        u=u,
        bounds=bounds,
        title="Utilities",
        cmap="Spectral",
        save=True,
        save_dir=utils_save_dir,
        filename="utilities",
        show_plot=True,
    )
else:
    utils_save_dir = ""

observer = noisy_observer(u=u, noise_variance=noise_variance, rng=rng)
init_X = rng.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(num_init_points, dims))
init_data = (init_X, observer(init_X))

models = create_models(data=init_data, kernel=kernel, noise_variance=noise_variance)

agent_dims_bounds = get_agent_dims_bounds(agent_dims=agent_dims)
acq_func = get_acquisition(
    acq_name=acq_name,
    beta=beta,
    bounds=bounds,
    agent_dims_bounds=agent_dims_bounds,
    mode=maxmin_mode,
    n_samples_outer=n_samples_outer,
)

ucb_funcs, lcb_funcs = create_ci_funcs(models=models, beta=beta)
for mode in ["DIRECT", "random"]:
    maxmin_mode = mode
    print("===============")
    print(f"maxmin_mode = {maxmin_mode}")

    if mode == "DIRECT":
        n_samples_outer = 20
    else:
        n_samples_outer = 100

    print(f"n_samples_outer: {n_samples_outer}")
    acq_func = get_acquisition(
        acq_name=acq_name,
        beta=beta,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode=maxmin_mode,
        n_samples_outer=n_samples_outer,
    )
    start = time.process_time()
    res = acq_func(models=models, rng=rng)
    end = time.process_time()
    print(f"res: {res}")
    res_score = evaluate_sample(
        res[0],
        outer_funcs=ucb_funcs,
        inner_funcs=lcb_funcs,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode="DIRECT",
        rng=rng,
    )
    print(f"no-regret sample score: {res_score}")
    print(f"Mode {mode} took {end-start} seconds")
