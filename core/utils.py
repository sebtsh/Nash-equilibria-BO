from itertools import chain, combinations
import numpy as np
from scipy.optimize import minimize


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


def join_action(s, i, actions):
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


def merge_data(data1, data2):
    """
    Joins two datasets represented as tuples (X, Y).
    :param data1: Tuple of 2 arrays of shape (n, N).
    :param data2: Tuple of 2 arrays of shape (m, N).
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


def all_equal_except_i(s1, s2, i):
    N = len(s1)
    is_all_equal = True
    for j in range(N):
        if j != i and s1[j] != s2[j]:
            is_all_equal = False
    return is_all_equal


def create_response_dict(domain, i):
    """
    Creates a dictionary with keys that are the bytes of a length N array, and returns the idxs of domain that have
    the actions of all other agents except i the same.
    :param domain: array of shape (M ** N, N).
    :param i: int.
    :return: dict.
    """
    _, N = domain.shape
    dic = {}
    for s in domain:
        idxs = []
        for idx, t in enumerate(domain):
            if all_equal_except_i(s, t, i):
                idxs.append(idx)
        dic[s.tobytes()] = idxs
    return dic


def unif_in_simplex(n, rng):
    k = rng.exponential(scale=1.0, size=n)
    return k / sum(k)


def sort_size_balance(pairs):
    inter = sorted(pairs, key=lambda x: x[0] + x[1])
    return sorted(inter, key=lambda x: abs(x[0] - x[1]))


def all_subset_actions(actions):
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    return list(powerset(actions))[1:]


def maximize_fn(f, bounds, rng, n_warmup=10000, n_iter=10):
    """
    Approximately maximizes a function f by sampling n_warmup random points in the domain, and then running L-BFGS-B
    starting from n_iter points. Adapted from https://github.com/fmfn/BayesianOptimization.
    :param f: Callable that takes in an array of shape (n, d) and returns an array of shape (n, 1).
    :param bounds: Array of shape (d, 2). Lower and upper bounds of each variable.
    :param rng: NumPy rng object.
    :param n_warmup: int. Number of random samples.
    :param n_iter: int. Number of L-BFGS-B starting points.
    :return: Array of shape (d,).
    """
    d = bounds.shape[0]
    # Random sampling
    x_tries = rng.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(n_warmup, d))
    f_x = f(x_tries)
    x_max = x_tries[np.argmax(f_x)]
    f_max = np.max(f_x)

    # L-BFGS-B
    x_seeds = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_iter - 1, d))
    x_seeds = np.concatenate([x_seeds, x_max[None, :]], axis=0)
    for x_try in x_seeds:
        res = minimize(
            fun=lambda x: np.squeeze(-f(x[None, :])),
            x0=x_try,
            bounds=bounds,
            method="L-BFGS-B",
        )
        if not res.success:
            continue
        if -res.fun >= f_max:
            x_max = res.x
            f_max = -res.fun

    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
