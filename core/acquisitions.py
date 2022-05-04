import numpy as np

from core.pne import best_response_payoff_pure, ucb_f
from core.mne import SEM_var_utility, get_strategies_and_support
from core.utils import cross_product, maximize_fn, maxmin_fn


def get_acquisition(
    acq_name,
    beta,
    bounds=None,
    agent_dims_bounds=None,
    mode=None,
    domain=None,
    actions=None,
    n_samples_outer=None,
):
    if acq_name == "ucb_pne":
        if bounds is None or agent_dims_bounds is None or mode is None or n_samples_outer is None:
            raise Exception("one of required params is None")
        return ucb_pne(
            beta=beta, bounds=bounds, agent_dims_bounds=agent_dims_bounds, mode=mode, n_samples_outer=n_samples_outer
        )
    elif acq_name == "ucb_mne":
        if domain is None or actions is None:
            raise Exception("domain or actions cannot be None")
        return ucb_mne(beta=beta, domain=domain, M=len(actions))
    else:
        raise Exception("Invalid acquisition name")


def create_ci_funcs(models, beta):
    """
    Converts GP models into UCB functions and LCB functions.
    :param models: List of N GPflow GPs.
    :param beta: float.
    :return: Tuple, 2 lists of Callables that take in an array of shape (n, dims) and return an array of shape (n, 1).
    """
    N = len(models)

    def create_ci_func(model, is_ucb):
        def inn(X):
            mean, var = model.predict_f(X)
            if is_ucb:
                return mean + beta * np.sqrt(var)
            else:  # is lcb
                return mean - beta * np.sqrt(var)

        return inn

    return (
        [create_ci_func(models[i], is_ucb=True) for i in range(N)],
        [create_ci_func(models[i], is_ucb=False) for i in range(N)],
    )


def ucb_pne_naive(beta, domain, actions, response_dicts):
    def acq(models):
        """
        Returns a point to query next.
        :param models: List of N GPflow GPs.
        :return: array of shape (1, N).
        """
        ucb_funcs, _ = create_ci_funcs(models=models, beta=beta)
        ucb_brp = best_response_payoff_pure(
            u=ucb_funcs, S=domain, actions=actions, response_dicts=response_dicts
        )  # array of shape (M ** N, N)

        next_idx = np.argmin(np.max(ucb_brp, axis=-1))
        return domain[next_idx : next_idx + 1]

    return acq


def ucb_pne_discrete(beta, domain, actions, response_dicts):
    def acq(models):
        """
        Returns N + 1 points to query next. First one is no-regret selection, next N are exploring samples.
        :param models: List of N GPflow GPs.
        :return: array of shape (N + 1, N).
        """
        M = len(actions)
        _, N = domain.shape
        ucb_funcs, lcb_funcs = create_ci_funcs(models=models, beta=beta)
        all_ucb = np.zeros((M**N, N))
        all_lcb = np.zeros((M**N, N))
        for j in range(N):
            all_ucb[:, j] = np.squeeze(ucb_funcs[j](domain), axis=-1)
            all_lcb[:, j] = np.squeeze(lcb_funcs[j](domain), axis=-1)

        samples_idxs = []
        # Pick no-regret selection
        ucb_f_vals = ucb_f(
            all_ucb=all_ucb,
            all_lcb=all_lcb,
            S=domain,
            actions=actions,
            response_dicts=response_dicts,
        )
        noreg_idx = np.argmax(np.min(ucb_f_vals, axis=-1))
        samples_idxs.append(noreg_idx)
        s_t = domain[noreg_idx]  # (N, )

        # Pick exploring samples
        for i in range(N):
            idxs = response_dicts[i][s_t.tobytes()]
            ucb_vals = all_ucb[idxs, i]
            samples_idxs.append(idxs[np.argmax(ucb_vals)])

        return domain[np.array(samples_idxs)]

    return acq


def maximize_ucb_f(
    ucb_funcs, lcb_funcs, bounds, agent_dims_bounds, rng, n_samples_outer=5
):
    """
    Computes argmax_s min_i ucb f(s) over a continuous domain.
    :param ucb_funcs: List of N Callables that take in an array of shape (n, dims) and return an array of shape (n, 1).
    :param lcb_funcs: List of N Callables that take in an array of shape (n, dims) and return an array of shape (n, 1).
    :param bounds: array of shape (dims, 2).
    :param agent_dims_bounds: List of N tuples (start_dim, end_dim).
    :param rng: NumPy rng object.
    :param n_samples_outer int.
    :return: Array of shape (dims).
    """
    dims = len(bounds)
    N = len(agent_dims_bounds)
    samples = rng.uniform(
        low=bounds[:, 0], high=bounds[:, 1], size=(n_samples_outer, dims)
    )

    # Obtain UCB values of utilities
    ucb_vals = np.concatenate(
        [ucb_funcs[i](samples) for i in range(N)], axis=-1
    )  # (num_samples_outer, N)

    # Obtain maximum LCB values of best response
    max_lcb_vals = []
    for s in samples:
        print("maximize_ucb_f sample computing")
        agent_max_lcb_vals = []
        for i in range(N):
            start_dim, end_dim = agent_dims_bounds[i]
            s_before = s[:start_dim]
            s_after = s[end_dim:]

            lcb_func = lcb_funcs[i]
            _, max_lcb_val = maximize_fn(
                f=lambda x: lcb_func(
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
            agent_max_lcb_vals.append(max_lcb_val)
        max_lcb_vals.append(agent_max_lcb_vals)
    max_lcb_vals = np.array(max_lcb_vals)  # (num_samples_outer, N)
    assert np.allclose(max_lcb_vals.shape, ucb_vals.shape)

    ucb_f_vals = np.minimum(ucb_vals - max_lcb_vals, 0.0)
    max_idx = np.argmax(np.min(ucb_f_vals, axis=-1))
    return samples[max_idx]


def ucb_pne(beta, bounds, agent_dims_bounds, mode, n_samples_outer):
    def acq(models, rng):
        """
        Returns N + 1 points to query next. First one is no-regret selection, next N are exploring samples.
        :param models: List of N GPflow GPs.
        :return: array of shape (N + 1, N).
        """
        N = len(agent_dims_bounds)
        ucb_funcs, lcb_funcs = create_ci_funcs(models=models, beta=beta)
        samples = []
        # Pick no-regret selection
        noreg_sample, _ = maxmin_fn(
            outer_funcs=ucb_funcs,
            inner_funcs=lcb_funcs,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            rng=rng,
            n_samples_outer=n_samples_outer,
        )
        samples.append(noreg_sample)

        # Pick exploring samples
        for i in range(N):
            start_dim, end_dim = agent_dims_bounds[i]
            s_before = noreg_sample[:start_dim]
            s_after = noreg_sample[end_dim:]

            ucb_func = ucb_funcs[i]
            max_ucb_sample, _ = maximize_fn(
                f=lambda x: np.squeeze(
                    ucb_func(
                        np.concatenate(
                            [
                                np.tile(s_before, (len(x), 1)),
                                x,
                                np.tile(s_after, (len(x), 1)),
                            ],
                            axis=-1,
                        )
                    ),
                ),
                bounds=bounds[start_dim:end_dim],
                rng=rng,
                n_warmup=100,
                n_iter=5,
            )
            samples.append(np.concatenate([s_before, max_ucb_sample, s_after]))

        return np.array(samples)

    return acq


def ucb_mne(beta, domain, M):
    def acq(models, prev_successes, rng):
        """
        Returns a pair of mixed strategies, and a batch of points to query next. Size of the batch will depend on the
        size of the supports of the potential MNE found.
        :param models: List of N GPflow GPs.
        :param prev_successes:
        :return: Tuple (tuple of 2 arrays of shape (M,), array of shape (B, N))
        """
        ucb_funcs, lcb_funcs = create_ci_funcs(models=models, beta=beta)
        U1upper = np.reshape(ucb_funcs[0](domain), (M, M))
        U1lower = np.reshape(lcb_funcs[0](domain), (M, M))
        U2upper = np.reshape(ucb_funcs[1](domain), (M, M))
        U2lower = np.reshape(lcb_funcs[1](domain), (M, M))

        mne_list, prev_successes = SEM_var_utility(
            U1upper=U1upper,
            U1lower=U1lower,
            U2upper=U2upper,
            U2lower=U2lower,
            num_rand_dists_per_agent=5,
            rng=rng,
            mode="first",
            prev_successes=prev_successes,
            evaluation_mode="linear_with_sampling",
            num_samples=10,
        )
        mne = mne_list[0]
        (s1, s2), (a1supp, a2supp) = get_strategies_and_support(mne, M, M)

        samples_coords = cross_product(a1supp[:, None], a2supp[:, None])
        print(f"samples: {samples_coords}")
        exploring_samples_coords = []
        for a1 in a1supp:
            a1_ucb = U2upper[a1]  # Given a1, can agent 2 do better?
            a1_ucb_argmax_a2coord = np.argmax(a1_ucb)
            if (
                a1_ucb_argmax_a2coord not in a2supp
            ):  # if it is, we would already have sampled this
                exploring_samples_coords.append([a1, a1_ucb_argmax_a2coord])
        for a2 in a2supp:
            a2_ucb = U1upper[:, a2]  # Given a2, can agent 1 do better?
            a2_ucb_argmax_a1coord = np.argmax(a2_ucb)
            if (
                a2_ucb_argmax_a1coord not in a1supp
            ):  # if it is, we would already have sampled this
                exploring_samples_coords.append([a2_ucb_argmax_a1coord, a2])

        print(f"exploring samples: {exploring_samples_coords}")
        if len(exploring_samples_coords) != 0:
            all_coords = np.concatenate(
                [samples_coords, np.array(exploring_samples_coords)], axis=0
            )  # (B, 2)
        else:
            all_coords = samples_coords
        all_domain_idxs = all_coords[:, 0] * M + all_coords[:, 1]

        return domain[all_domain_idxs], (s1, s2), prev_successes

    return acq
