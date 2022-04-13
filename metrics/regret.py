import numpy as np

from core.pne import best_response_payoff_pure
from core.mne import neg_brp_mixed
from core.utils import arr_index


def calc_regret_pne(u, data, domain, actions, response_dicts):
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
