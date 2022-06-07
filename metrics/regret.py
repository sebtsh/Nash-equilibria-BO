import numpy as np

from core.pne import best_response_payoff_pure_discrete
from core.mne import neg_brp_mixed
from core.utils import arr_index, maxmin_fn
from core.pne import evaluate_sample
from core.models import create_models, create_mean_funcs


def calc_regret_pne(
    u, data, bounds, agent_dims_bounds, mode, rng, n_samples_outer, known_best_val
):
    if known_best_val is None:
        _, best_val = maxmin_fn(
            outer_funcs=u,
            inner_funcs=u,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            rng=rng,
            n_samples_outer=200,
        )
    else:
        best_val = known_best_val

    X, _ = data
    sample_regret = []
    cumu_regret = []

    if mode == "DIRECT":
        maximize_mode = "DIRECT"
    elif mode == "random":
        maximize_mode = "L-BFGS-B"
    else:
        raise Exception("Incorrect mode passed to calc_regret_pne")
    for x in X:
        x_val = evaluate_sample(
            s=x,
            outer_funcs=u,
            inner_funcs=u,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            rng=rng,
            mode=maximize_mode,
        )
        sample_regret.append(np.maximum(best_val - x_val, 0.))
        cumu_regret.append(np.sum(sample_regret))

    return np.array(sample_regret), np.array(cumu_regret)


def calc_regret_pne_discrete(u, data, domain, response_dicts):
    X, _ = data
    brp = best_response_payoff_pure_discrete(
        u=u, domain=domain, response_dicts=response_dicts
    )  # (M ** N, N)
    strategy_eps = np.max(brp, axis=-1)
    best = np.min(strategy_eps)

    sample_regret = []
    cumu_regret = []
    for x in X:
        idx = arr_index(domain, x)
        sample_regret.append(strategy_eps[idx] - best)
        cumu_regret.append(np.sum(sample_regret))

    return np.array(sample_regret), np.array(cumu_regret)


def calc_imm_regret_pne(
    u,
    data,
    num_agents,
    num_init_points,
    kernel,
    noise_variance,
    bounds,
    agent_dims_bounds,
    mode,
    rng,
    n_samples_outer,
    known_best_val,
    skip_length,
):
    if known_best_val is None:
        _, best_val = maxmin_fn(
            outer_funcs=u,
            inner_funcs=u,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            rng=rng,
            n_samples_outer=n_samples_outer,
        )
    else:
        best_val = known_best_val

    imm_regret = []
    if mode == "DIRECT":
        maximize_mode = "DIRECT"
    elif mode == "random":
        maximize_mode = "L-BFGS-B"
    else:
        raise Exception("Incorrect mode passed to calc_imm_regret_pne")
    for t in np.arange(0, len(data[0]) - num_init_points, skip_length):
        models = create_models(
            num_agents=num_agents,
            data=data[: t + num_init_points + 1],
            kernel=kernel,
            noise_variance=noise_variance,
        )
        mean_funcs = create_mean_funcs(models=models)
        guessed_ne, _ = maxmin_fn(
            outer_funcs=mean_funcs,
            inner_funcs=mean_funcs,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            rng=rng,
            n_samples_outer=n_samples_outer,
        )
        x_val = evaluate_sample(
            s=guessed_ne,
            outer_funcs=u,
            inner_funcs=u,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            rng=rng,
            mode=maximize_mode,
        )
        imm_regret.append(best_val - x_val)

    return np.array(imm_regret)


def calc_regret_mne(strategies, U1, U2):
    sample_regret = []
    cumu_regret = []
    for s in strategies:
        reg = -np.min(neg_brp_mixed(U1, U2, s))
        sample_regret.append(reg)
        cumu_regret.append(np.sum(sample_regret))

    return np.array(sample_regret), np.array(cumu_regret)
