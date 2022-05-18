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
    create_response_dict,
    cross_product,
    sobol_sequence,
)
from core.pne import find_PNE_discrete, best_response_payoff_pure_discrete
from core.models import create_models
from metrics.regret import calc_regret_pne
from metrics.plotting import plot_utilities_2d, plot_utilities_2d_discrete, plot_regret
from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path

utility_name = "rand"
acq_name = "ucb_pne"
agent_dims = [2, 2]  # this determines num_agents and dims
ls = np.array([0.5] * sum(agent_dims))
bound = [-1.0, 1.0]  # assumes same bounds for all dims
noise_variance = 0.001
num_init_points = 10
num_iters = 800
beta = 2.0
maxmin_mode = "DIRECT"
n_samples_outer = 10
seed = 0
known_best_val = None
num_actions_discrete = (
    64  # for acquisition functions that require discretization, i.e. prob_eq and SUR
)

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

# Just for this discrete test, the init_X must be sampled from domain
observer = noisy_observer(u=u, noise_variance=noise_variance, rng=rng)
init_idxs = rng.integers(low=0, high=len(domain), size=num_init_points)
init_X = domain[init_idxs]
init_data = (init_X, observer(init_X))

print("Finding PNE")
brp = best_response_payoff_pure_discrete(
    u=u, domain=domain, num_actions=num_actions_discrete, response_dicts=response_dicts
)
pne, idx = find_PNE_discrete(
    u=u, domain=domain, num_actions=num_actions_discrete, response_dicts=response_dicts
)
print(f"PNE at {pne} with idx {idx} and brp {brp[np.argmin(np.max(brp, axis=-1))]}")

models = create_models(
    num_agents=num_agents, data=init_data, kernel=kernel, noise_variance=noise_variance
)
from core.acquisitions import prob_eq

print("Calculating prob_eq")
prob_eq_vals = prob_eq(
    models=models,
    domain=domain,
    response_dicts=response_dicts,
    num_actions=num_actions_discrete,
)
print(prob_eq_vals)
print(
    f"Strategy profile with max prob_eq: {domain[np.argmax(prob_eq_vals)]}, with a prob_eq value of {np.max(prob_eq_vals)}, idx {np.argmax(prob_eq_vals)}"
)
print(f"Chosen strategy profile has a brp of {brp[np.argmax(prob_eq_vals)]}")
print(f"True NE has a prob_eq value of {prob_eq_vals[np.argmin(np.max(brp, axis=-1))]}")
