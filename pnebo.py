import numpy as np
import tensorflow as tf
import gpflow as gpf
from core.objectives import sample_GP_prior_utilities, noisy_observer
from core.models import create_models
from core.utils import cross_product, create_response_dict
from core.optimization import bo_loop
from core.acquisitions import get_acquisition
from metrics.regret import calc_regret
from metrics.plotting import plot_utilities_2d, plot_regret

# Parameters
acq_name = "ucb_pne"  # 'ucb_pne_naive', 'ucb_pne'
seed = 0
num_agents = 3
num_actions = 8
ls = np.array([0.2] * num_agents)
lowers = [0.0] * num_agents
uppers = [1.0] * num_agents
noise_variance = 0.1
num_init_points = 2
num_iters = 800
beta = 2.0
plot_utils = False
dir = "results/test/"
rng = np.random.default_rng(seed)
tf.random.set_seed(seed)

actions = np.linspace(lowers[0], uppers[0], num_actions)
domain = actions[:, None]
for i in range(1, num_agents):
    next_actions = np.linspace(lowers[i], uppers[i], num_actions)
    domain = cross_product(domain, next_actions[:, None])
response_dicts = [create_response_dict(domain, i) for i in range(num_agents)]

kernel = gpf.kernels.SquaredExponential(lengthscales=ls)
u = sample_GP_prior_utilities(
    num_agents, kernel, lowers, uppers, num_points=100, rng=rng
)
if plot_utils:
    utils_save_dir = dir + "gif/"
    plot_utilities_2d(
        u=u,
        xlims=(lowers[0], uppers[0]),
        ylims=(lowers[1], uppers[1]),
        actions=actions,
        domain=domain,
        response_dicts=response_dicts,
        title="Utilities",
        cmap="Spectral",
        save=True,
        save_dir=utils_save_dir,
        filename="utilities",
        show_plot=False,
    )
else:
    utils_save_dir = ""

observer = noisy_observer(u=u, noise_variance=noise_variance, rng=rng)
init_idxs = rng.integers(low=0, high=num_actions**num_agents, size=num_init_points)
init_X = domain[init_idxs]
init_data = (init_X, observer(init_X))

models = create_models(data=init_data, kernel=kernel, noise_variance=noise_variance)

acq_func = get_acquisition(
    acq_name=acq_name,
    beta=beta,
    domain=domain,
    actions=actions,
    response_dicts=response_dicts,
)

final_data = bo_loop(
    init_data=init_data,
    observer=observer,
    models=models,
    acquisition=acq_func,
    num_iters=num_iters,
    kernel=kernel,
    noise_variance=noise_variance,
    actions=actions,
    domain=domain,
    plot=plot_utils,
    save_dir=utils_save_dir,
)

final_data_minus_init = (
    final_data[0][num_init_points:],
    final_data[1][num_init_points:],
)
imm_regret, cumu_regret = calc_regret(
    u=u,
    data=final_data_minus_init,
    domain=domain,
    actions=actions,
    response_dicts=response_dicts,
)

regrets_save_dir = dir + "regrets/"

plot_regret(
    regret=imm_regret,
    num_iters=num_iters * (num_agents + 1),
    title="Immediate regret (all samples)",
    save=True,
    save_dir=regrets_save_dir,
    filename="imm-all",
)
# plot_regret(regret=cumu_regret,
#             num_iters=num_iters * (num_agents + 1),
#             title='Cumulative regret (all samples)',
#             save=True,
#             save_dir=regrets_save_dir,
#             filename='cumu-all')

noreg_seq = np.array([j * (num_agents + 1) for j in range(num_iters)], dtype=np.int32)
plot_regret(
    regret=imm_regret[noreg_seq],
    num_iters=num_iters,
    title="Immediate regret (no-regret sequence)",
    save=True,
    save_dir=regrets_save_dir,
    filename="imm-noregseq",
)
# plot_regret(regret=cumu_regret[noreg_seq],
#             num_iters=num_iters,
#             title='Cumulative regret (no-regret sequence)',
#             save=True,
#             save_dir=regrets_save_dir,
#             filename='cumu-noregseq')

# print("Regrets of all samples")
# print(imm_regret)
# print(cumu_regret)
# print("Regrets of no-regret sequence")
# print(imm_regret[noreg_seq])
# print(cumu_regret[noreg_seq])
