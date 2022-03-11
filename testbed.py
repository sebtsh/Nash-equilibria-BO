import numpy as np
import tensorflow as tf
import gpflow as gpf
from core.objectives import sample_GP_prior_utilities, noisy_observer
from core.models import create_models
from core.utils import cross_product
from core.optimization import bo_loop
from core.acquisitions import ucb_ne
from metrics.regret import calc_regret

# Parameters
seed = 0
rng = np.random.default_rng(seed)
tf.random.set_seed(seed)
num_agents = 2
num_actions = 20
ls = np.array([0.1] * num_agents)
lowers = [0.] * num_agents
uppers = [1.] * num_agents
noise_variance = 0.1
num_init_points = 3
num_bo_iters = 200
beta = 2.

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

acq_func = ucb_ne(beta=beta,
                  domain=domain,
                  actions=actions)

final_data = bo_loop(init_data=init_data,
                     observer=observer,
                     models=models,
                     acquisition=acq_func,
                     num_bo_iters=num_bo_iters,
                     kernel=kernel,
                     noise_variance=noise_variance)

final_data_minus_init = (final_data[0][num_init_points:], final_data[1][num_init_points:])
imm_regret, cumu_regret = calc_regret(u=u,
                                      data=final_data_minus_init,
                                      domain=domain,
                                      actions=actions)

print(imm_regret)
print(cumu_regret)
