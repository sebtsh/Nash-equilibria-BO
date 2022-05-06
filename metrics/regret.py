import numpy as np

from core.pne import best_response_payoff_pure
from core.mne import neg_brp_mixed
from core.utils import arr_index, maxmin_fn
from core.pne import evaluate_sample


def calc_regret_pne(
    u, data, bounds, agent_dims_bounds, mode, rng, n_samples_outer, known_best_val=None
):
    if known_best_val is not None:
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

    X, _ = data
    imm_regret = []
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
        imm_regret.append(best_val - x_val)
        cumu_regret.append(np.sum(imm_regret))

    return np.array(imm_regret), np.array(cumu_regret)


def calc_regret_pne_discrete(u, data, domain, actions, response_dicts):
    X, _ = data
    brp = best_response_payoff_pure(
        u=u, S=domain, actions=actions, response_dicts=response_dicts
    )  # (M ** N, N)
    strategy_eps = np.max(brp, axis=-1)
    best = np.min(strategy_eps)

    imm_regret = []
    cumu_regret = []
    for x in X:
        idx = arr_index(domain, x)
        imm_regret.append(strategy_eps[idx] - best)
        cumu_regret.append(np.sum(imm_regret))

    return np.array(imm_regret), np.array(cumu_regret)


def calc_regret_mne(strategies, U1, U2):
    imm_regret = []
    cumu_regret = []
    for s in strategies:
        reg = -np.min(neg_brp_mixed(U1, U2, s))
        imm_regret.append(reg)
        cumu_regret.append(np.sum(imm_regret))

    return np.array(imm_regret), np.array(cumu_regret)
