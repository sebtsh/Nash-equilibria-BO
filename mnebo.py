import numpy as np
import tensorflow as tf
import gpflow as gpf
import pickle
from core.objectives import get_utilities, noisy_observer
from core.utils import cross_product, sobol_sequence, get_agent_dims_bounds
from core.optimization import bo_loop_mne
from core.acquisitions import get_acq_mixed
from core.mne import SEM, neg_brp_mixed
from metrics.regret import calc_regret_mne
from metrics.plotting import plot_utilities_2d, plot_regret

# Parameters
acq_name = "ucb_mne"
utility_name = "rand"
seed = 0
num_actions = 16
agent_dims = [2, 2]  # this determines num_agents and dims
num_agents = len(agent_dims)
ls = np.array([0.5] * sum(agent_dims))
dims = np.sum(agent_dims)
bound = [-1.0, 1.0]  # assumes same bounds for all dims
noise_variance = 0.001
num_init_points = 2
num_iters = 200
beta = 2.0
plot_utils = False
dir = "results/mne/"
rng = np.random.default_rng(seed)
tf.random.set_seed(seed)

# Execution
bounds = np.array([bound for _ in range(dims)])
agent_dims_bounds = get_agent_dims_bounds(agent_dims=agent_dims)
start_dim, end_dim = agent_dims_bounds[0]
domain = sobol_sequence(num_points=num_actions, bounds=bounds[start_dim:end_dim])
for i in range(1, num_agents):
    start_dim, end_dim = agent_dims_bounds[i]
    domain = cross_product(
        domain, sobol_sequence(num_points=num_actions, bounds=bounds[start_dim:end_dim])
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
init_idxs = rng.integers(low=0, high=num_actions**num_agents, size=num_init_points)
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
    plot=plot_utils,
)

imm_regret, cumu_regret = calc_regret_mne(strategies=chosen_strategies, U1=U1, U2=U2)
print("Immediate regret:")
print(imm_regret)
print("Cumulative regret:")
print(cumu_regret)
regrets_save_dir = dir + "regrets/"
plot_regret(
    regret=imm_regret,
    num_iters=num_iters,
    title="MNE: Immediate regret",
    save=True,
    save_dir=regrets_save_dir,
    filename="mne-imm",
)
plot_regret(
    regret=cumu_regret,
    num_iters=num_iters,
    title="MNE: Cumulative regret",
    save=True,
    save_dir=regrets_save_dir,
    filename="mne-cumu",
)
