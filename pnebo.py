import numpy as np
import tensorflow as tf
import gpflow as gpf
import time
from core.objectives import sample_GP_prior_utilities, noisy_observer
from core.models import create_models
from core.optimization import bo_loop_pne
from core.acquisitions import get_acquisition
from core.utils import get_agent_dims_bounds
from metrics.regret import calc_regret_pne
from metrics.plotting import plot_utilities_2d, plot_regret

# Parameters
acq_name = "ucb_pne"  # 'ucb_pne_naive', 'ucb_pne'
seed = 0
agent_dims = [1, 1]  # this determines num_agents and dims
num_agents = len(agent_dims)
dims = np.sum(agent_dims)
ls = np.array([0.4] * dims)
bounds = np.array([[-1.0, 1.0] for _ in range(dims)])
noise_variance = 0.1
num_init_points = 10
num_iters = 50
beta = 2.0
maxmin_mode = "random"
plot_utils = True
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
)

for mode in ["random", "DIRECT"]:
    maxmin_mode = mode
    print("===============")
    print(f"maxmin_mode = {maxmin_mode}")
    acq_func = get_acquisition(
        acq_name=acq_name,
        beta=beta,
        bounds=bounds,
        agent_dims_bounds=agent_dims_bounds,
        mode=maxmin_mode,
    )
    start = time.process_time()
    res = acq_func(models=models, rng=rng)
    end = time.process_time()
    print(f"res: {res}")
    print(f"Mode {mode} took {end-start} seconds")


# final_data = bo_loop_pne(
#     init_data=init_data,
#     observer=observer,
#     models=models,
#     acquisition=acq_func,
#     num_iters=num_iters,
#     kernel=kernel,
#     noise_variance=noise_variance,
#     actions=actions,
#     domain=domain,
#     plot=plot_utils,
#     save_dir=utils_save_dir,
# )
#
# final_data_minus_init = (
#     final_data[0][num_init_points:],
#     final_data[1][num_init_points:],
# )
# imm_regret, cumu_regret = calc_regret_pne(
#     u=u,
#     data=final_data_minus_init,
#     domain=domain,
#     actions=actions,
#     response_dicts=response_dicts,
# )
#
# regrets_save_dir = dir + "regrets/"
#
# plot_regret(
#     regret=imm_regret,
#     num_iters=num_iters * (num_agents + 1),
#     title="Immediate regret (all samples)",
#     save=True,
#     save_dir=regrets_save_dir,
#     filename="imm-all",
# )
# # plot_regret(regret=cumu_regret,
# #             num_iters=num_iters * (num_agents + 1),
# #             title='Cumulative regret (all samples)',
# #             save=True,
# #             save_dir=regrets_save_dir,
# #             filename='cumu-all')
#
# noreg_seq = np.array([j * (num_agents + 1) for j in range(num_iters)], dtype=np.int32)
# plot_regret(
#     regret=imm_regret[noreg_seq],
#     num_iters=num_iters,
#     title="Immediate regret (no-regret sequence)",
#     save=True,
#     save_dir=regrets_save_dir,
#     filename="imm-noregseq",
# )
# plot_regret(
#     regret=cumu_regret[noreg_seq],
#     num_iters=num_iters,
#     title="Cumulative regret (no-regret sequence)",
#     save=True,
#     save_dir=regrets_save_dir,
#     filename="cumu-noregseq",
# )
#
# print("Regrets of all samples")
# print(imm_regret)
# print(cumu_regret)
# print("Regrets of no-regret sequence")
# print(imm_regret[noreg_seq])
# print(cumu_regret[noreg_seq])
