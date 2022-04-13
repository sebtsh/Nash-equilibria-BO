import numpy as np
import tensorflow as tf
import gpflow as gpf
from core.objectives import sample_GP_prior_utilities, noisy_observer
from core.utils import cross_product, create_response_dict
from core.optimization import bo_loop_mne
from core.acquisitions import get_acquisition
from core.mne import SEM, neg_brp_mixed
from metrics.regret import calc_regret_mne
from metrics.plotting import plot_utilities_2d, plot_regret

# Parameters
acq_name = "ucb_mne"  # 'ucb_pne_naive', 'ucb_pne', 'ucb_mne'
seed = 0
num_agents = 2
num_actions = 5
ls = np.array([0.5] * num_agents)
lowers = [0.0] * num_agents
uppers = [1.0] * num_agents
noise_variance = 0.1
num_init_points = 2
num_iters = 400
beta = 2.0
plot_utils = True
dir = "results/mne/"
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

U1 = np.reshape(u[0](domain), (num_actions, num_actions))
U2 = np.reshape(u[1](domain), (num_actions, num_actions))
print("U1:")
print(U1)
print("U2:")
print(U2)

all_res = SEM(U1, U2)
for res in all_res:
    s1 = res[:num_actions]
    s2 = res[num_actions + 1 : num_actions + num_actions + 1]
    print(f"Agent 1 strategy: {s1}, with value {res[num_actions]}")
    print(f"Agent 2 strategy: {s2}, with value {res[num_actions + num_actions + 1]}")
    print(f"-brp: {neg_brp_mixed(U1, U2, (s1, s2))}")
    print("==============")

observer = noisy_observer(u=u, noise_variance=noise_variance, rng=rng)
init_idxs = rng.integers(low=0, high=num_actions**num_agents, size=num_init_points)
init_X = domain[init_idxs]
init_data = (init_X, observer(init_X))

acq_func = get_acquisition(
    acq_name=acq_name,
    beta=beta,
    domain=domain,
    actions=actions,
    response_dicts=response_dicts,
    rng=rng,
)

final_data, chosen_strategies = bo_loop_mne(
    init_data=init_data,
    observer=observer,
    acquisition=acq_func,
    num_iters=num_iters,
    kernel=kernel,
    noise_variance=noise_variance,
    actions=actions,
    domain=domain,
    plot=plot_utils,
    save_dir=utils_save_dir,
)

imm_regret, cumu_regret = calc_regret_mne(strategies=chosen_strategies, U1=U1, U2=U2)
print("Immediate regret:")
print(imm_regret)
print("Cumulative regret:")
print(cumu_regret)
