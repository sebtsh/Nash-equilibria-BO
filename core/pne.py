import numpy as np
from core.utils import maximize_fn


def best_response_payoff_pure(u, S, actions, response_dicts):
    """
    Calculates the best response payoff for each pure strategy profile in S, for each agent. As currently implemented,
    O(M^2N^2) operation.
    :param u: List of utility functions.
    :param S: array of shape (M ** N, N). All possible pure strategy profiles of the N agents.
    :param actions: array of shape (M, ). All possible M actions.
    :param response_dicts: list of N dictionaries.
    :return: array of shape (M ** N, N).
    """
    M = len(actions)
    _, N = S.shape
    brp = np.zeros((M**N, N))
    all_utils = np.zeros((M**N, N))
    for j in range(N):
        all_utils[:, j] = np.squeeze(u[j](S), axis=-1)

    for i in range(len(S)):
        s = S[i]
        for j in range(N):
            idxs = response_dicts[j][s.tobytes()]
            utils = all_utils[idxs, j]
            best_util = np.max(utils)
            current_util = all_utils[i, j]
            brp[i, j] = best_util - current_util

    return brp


def ucb_f(all_ucb, all_lcb, S, actions, response_dicts):
    """
    Calculates the upper confidence bound of the negative best response payoff for each pure strategy profile in S, for
    each agent.
    :param all_ucb: array of shape (M ** N, N).
    :param all_lcb: array of shape (M ** N, N).
    :param S: array of shape (M ** N, N). All possible pure strategy profiles of the N agents.
    :param actions: array of shape (M, ). All possible M actions.
    :param response_dicts: list of N dictionaries.
    :return: array of shape (M ** N, N).
    """
    M = len(actions)
    _, N = S.shape
    ucb_f_vals = np.zeros((M**N, N))

    for i in range(len(S)):
        s = S[i]
        for j in range(N):
            idxs = response_dicts[j][s.tobytes()]
            lcb_vals = all_lcb[idxs, j]
            max_lcb = np.max(lcb_vals)
            current_ucb = all_ucb[i, j]
            ucb_f_vals[i, j] = current_ucb - max_lcb

    return np.maximum(ucb_f_vals, 0)  # since f <= 0
