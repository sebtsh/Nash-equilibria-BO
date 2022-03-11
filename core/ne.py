import numpy as np

from core.utils import join_action


def best_response_payoff_pure(u,
                              S,
                              actions):
    """
    Calculates the best response payoff for each pure strategy profile in S, for each agent. As currently implemented,
    O(M^2N^2) operation.
    :param u: List of utility functions.
    :param S: array of shape (M ** N, N). All possible pure strategy profiles of the N agents.
    :param actions: array of shape (M, ). All possible M actions.
    :return: array of shape (M ** N, N).
    """
    M = len(actions)
    _, N = S.shape
    brp = np.zeros((M ** N, N))

    for i in range(N):
        for j in range(len(S)):
            s = S[j]
            joined = join_action(s, i, actions)
            best_util = np.max(u[i](joined))
            current_util = u[i](s[None, :])
            brp[j, i] = best_util - current_util

    return brp
