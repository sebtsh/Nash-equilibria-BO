import numpy as np
from core.utils import maximize_fn


def best_response_payoff_pure_discrete(u, domain, response_dicts, is_u_func=True):
    """
    Calculates the best response payoff for each pure strategy profile in S, for each agent. As currently implemented,
    O(M^2N^2) operation. WARNING: Assumes all agents have the same number of actions.
    :param u: List of utility functions OR domain utility values as an array of shape (M ** N, N). Indicate which it is
    with is_u_func.
    :param domain: array of shape (M ** N, N). All possible pure strategy profiles of the N agents.
    :param response_dicts: list of N dictionaries.
    :param is_u_func: True if u is a list of utility functions and False if u is utility array of shape (M ** N, N).
    :return: array of shape (M ** N, N).
    """
    N = len(response_dicts)
    brp = np.zeros((len(domain), N))

    if is_u_func:
        all_utils = np.zeros((len(domain), N))
        for j in range(N):
            all_utils[:, j] = np.squeeze(u[j](domain), axis=-1)
    else:
        all_utils = u

    for i, s in enumerate(domain):
        for j in range(N):
            idxs = response_dicts[j][s.tobytes()]
            utils = all_utils[idxs, j]
            best_util = np.max(utils)
            current_util = all_utils[i, j]
            brp[i, j] = best_util - current_util

    return brp


def evaluate_sample(s, outer_funcs, inner_funcs, bounds, agent_dims_bounds, rng, mode):
    N = len(agent_dims_bounds)

    outer_vals = np.array(
        [np.squeeze(outer_funcs[i](s[None, :])) for i in range(N)]
    )  # (N)

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
            mode=mode,
            n_warmup=100,
            n_iter=5,
        )
        agent_max_inner_vals.append(max_inner_val)
    return np.min(outer_vals - np.array(agent_max_inner_vals))
