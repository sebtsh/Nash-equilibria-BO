import numpy as np
import tensorflow as tf
import gpflow as gpf
from core.objectives import sample_GP_prior_utilities, noisy_observer
from core.models import create_models
from core.utils import cross_product


seed = 0
rng = np.random.default_rng(seed)
tf.random.set_seed(seed)
num_agents = 2
num_actions = 8
ls = np.array([0.1] * num_agents)
lowers = [0.] * num_agents
uppers = [1.] * num_agents
noise_variance = 0.1
num_init_points = 3

actions = np.linspace(lowers[0], uppers[0], num_actions)
domain = actions[:, None]
for i in range(1, num_agents):
    next_actions = np.linspace(lowers[i], uppers[i], num_actions)
    domain = cross_product(domain, next_actions[:, None])

kernel = gpf.kernels.SquaredExponential(lengthscales=ls)
u = sample_GP_prior_utilities(num_agents,
                              kernel,
                              lowers,
                              uppers,
                              num_points=100,
                              rng=rng)
observer = noisy_observer(u=u,
                          noise_variance=noise_variance,
                          rng=rng)
init_idxs = rng.integers(low=0, high=num_actions ** num_agents, size=num_init_points)
init_X = domain[init_idxs]
init_data = (init_X, observer(init_X))

models = create_models(data=init_data,
                       kernel=kernel,
                       noise_variance=noise_variance)
