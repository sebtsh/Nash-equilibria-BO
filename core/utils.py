from itertools import chain, combinations
import numpy as np
from scipy.optimize import minimize
from scipydirect import minimize as direct_minimize


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
    Approximately maximizes a function f using sampling + L-BFGS-B method adapted from
    https://github.com/fmfn/BayesianOptimization.
    :param f: Callable that takes in an array of shape (n, d) and returns an array of shape (n, 1).
    :param bounds: Array of shape (d, 2). Lower and upper bounds of each variable.
    :param rng: NumPy rng object.
    :param mode: str. Either 'DIRECT' or 'L-BFGS-B'.
    :param n_warmup: int. Number of random samples.
    :param n_iter: int. Number of L-BFGS-B starting points.
    :return: Array of shape (d,).
    """
    d = len(bounds)

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

    return np.clip(x_max, bounds[:, 0], bounds[:, 1]), f_max


def get_agent_dims_bounds(agent_dims):
    """
     WARNING FOR FUTURE SELF: To get agent i's slice of a strategy profile, do s[start_dim : end_dim], i.e., the +1 is
     included here. This was not the case for the mixed NE code.
    :param agent_dims: list of N ints.
    :return: returns a list of N tuples (start_dim, end_dim) for each agent.
    """
    current_start = 0
    agent_dims_bounds = []
    for interval in agent_dims:
        next_start = current_start + interval
        agent_dims_bounds.append((current_start, next_start))
        current_start = next_start
    return agent_dims_bounds


def maxmin_fn(
    outer_funcs,
    inner_funcs,
    bounds,
    agent_dims_bounds,
    mode,
    rng,
    n_samples_outer=100,
):
    """

    :param outer_funcs:
    :param inner_funcs:
    :param bounds:
    :param agent_dims_bounds:
    :param rng:
    :param mode: str. Either 'DIRECT' or 'random'.
    :param n_samples_outer:
    :return:
    """
    dims = len(bounds)
    N = len(agent_dims_bounds)

    if mode == "random":
        samples = rng.uniform(
            low=bounds[:, 0], high=bounds[:, 1], size=(n_samples_outer, dims)
        )

        # Obtain values of outer function
        outer_vals = np.concatenate(
            [outer_funcs[i](samples) for i in range(N)], axis=-1
        )  # (num_samples_outer, N)

        # Obtain maximum of inner function
        max_inner_vals = []
        for s in samples:
            # print("maximize_ucb_f sample computing")
            agent_max_inner_vals = []
            for i in range(N):
                start_dim, end_dim = agent_dims_bounds[i]
                s_before = s[:start_dim]
                s_after = s[end_dim:]

                inner_func = inner_funcs[i]
                _, max_inner_val = maximize_fn(
                    f=lambda x: inner_func(
                        np.concatenate(
                            [
                                np.tile(s_before, (len(x), 1)),
                                x,
                                np.tile(s_after, (len(x), 1)),
                            ],
                            axis=-1,
                        )
                    ),
                    bounds=bounds[start_dim:end_dim],
                    rng=rng,
                    n_warmup=100,
                    n_iter=5,
                )
                agent_max_inner_vals.append(max_inner_val)
            max_inner_vals.append(agent_max_inner_vals)
        max_inner_vals = np.array(max_inner_vals)  # (num_samples_outer, N)
        assert np.allclose(max_inner_vals.shape, outer_vals.shape)

        outer_minus_inner_vals = np.minimum(outer_vals - max_inner_vals, 0.0)
        max_idx = np.argmax(np.min(outer_minus_inner_vals, axis=-1))
        max_val = np.min(outer_minus_inner_vals, axis=-1)[max_idx]

        return samples[max_idx], max_val

    elif mode == "DIRECT":
        def obj(s):
            print("Calling inner obj")
            agent_max_inner_vals = []
            for i in range(N):
                start_dim, end_dim = agent_dims_bounds[i]
                s_before = s[:start_dim]
                s_after = s[end_dim:]

                inner_func = inner_funcs[i]
                _, max_inner_val = maximize_fn(
                    f=lambda x: inner_func(
                        np.concatenate(
                            [
                                np.tile(s_before, (len(x), 1)),
                                x,
                                np.tile(s_after, (len(x), 1)),
                            ],
                            axis=-1,
                        )
                    ),
                    bounds=bounds[start_dim:end_dim],
                    rng=rng,
                    n_warmup=100,
                    n_iter=5,
                )
                agent_max_inner_vals.append(max_inner_val)
            outer_vals = np.array(
                [np.squeeze(outer_funcs[i](s[None, :])) for i in range(N)]
            )
            print("Finished inner obj")
            return np.max(np.array(agent_max_inner_vals) - outer_vals)

        res = direct_minimize(obj, bounds=bounds, algmethod=1, maxT=n_samples_outer)

        return res.x, res.fun

    else:
        raise Exception("Incorrect mode passed to maxmin_fn")
