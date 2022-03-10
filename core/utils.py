import numpy as np


def combinations(points1, points2):
    """
    :return: tensor of shape (n ** 2, d * 2)
    """
    n = points1.shape[0]
    d = points1.shape[1]

    out = np.zeros((n * n, d * 2))
    for i in range(n):
        for j in range(n):
            out[i * n + j][0:d] = points1[i]
            out[i * n + j][d:d * 2] = points2[j]
    return out


def join_action(s,
                i,
                actions):
    """
    Given a strategy profile s, fixes the actions of all other agents, and replaces the action of the i-th agent
    with all possible actions.
    :param s: Array of shape (N, ). Strategy profile.
    :param i: int. Agent.
    :param actions: Array of shape (M, ). All possible actions.
    :return: Array of shape (M, N).
    """
    M = len(actions)
    joined = np.tile(s[None, :], (M, 1))
    joined[:, i] = actions
    return joined
