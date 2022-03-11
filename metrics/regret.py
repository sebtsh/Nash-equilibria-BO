import numpy as np

from core.ne import best_response_payoff_pure
from core.utils import arr_index


def calc_regret(u,
                data,
                domain,
                actions):
    X, _ = data
    brp = best_response_payoff_pure(u=u,
                                    S=domain,
                                    actions=actions)  # (M ** N, N)
    strategy_eps = np.max(brp, axis=-1)
    best = np.min(strategy_eps)

    imm_regret = []
    cumu_regret = []
    for x in X:
        idx = arr_index(domain, x)
        imm_regret.append(strategy_eps[idx] - best)
        cumu_regret.append(np.sum(imm_regret))

    return imm_regret, cumu_regret
