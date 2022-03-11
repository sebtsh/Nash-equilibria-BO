import numpy as np


def cross_product(x, y):
    """

    :param x: array of shape (m, d_x)
    :param y: array of shape (n, d_y)
    :return:  array of shape (m * n, d_x + d_y)
    """
    m, d_x = x.shape
    n, d_y = y.shape
    x_temp = np.tile(x[:, :, None], (1, n, 1))
    x_temp = np.reshape(x_temp, [m * n, d_x])
    y_temp = np.tile(y, (m, 1))
    return np.concatenate([x_temp, y_temp], axis=-1)


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


def merge_data(data1,
               data2):
    """
    Joins two datasets represented as tuples (X, Y).
    :param data1: Array of shape (n, N).
    :param data2: Array of shape (m, N).
    :return: Array of shape (n + m, N).
    """
    X_1, Y_1 = data1
    X_2, Y_2 = data2
    X_new = np.concatenate((X_1, X_2), axis=0)
    Y_new = np.concatenate((Y_1, Y_2), axis=0)
    return X_new, Y_new


def arr_index(array, item):
    for idx, val in enumerate(array):
        if np.allclose(val, item):
            return idx
