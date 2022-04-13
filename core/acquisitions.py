import numpy as np

from core.pne import best_response_payoff_pure, ucb_f
from core.mne import SEM_var_utility, get_strategies_and_support
from core.utils import cross_product


def get_acquisition(acq_name, beta, domain, actions, response_dicts, rng):
    if acq_name == "ucb_pne_naive":
        return ucb_pne_naive(
            beta=beta, domain=domain, actions=actions, response_dicts=response_dicts
        )
    elif acq_name == "ucb_pne":
        return ucb_pne(
            beta=beta, domain=domain, actions=actions, response_dicts=response_dicts
        )
    elif acq_name == "ucb_mne":
        return ucb_mne(beta=beta, domain=domain, M=len(actions), rng=rng)
    else:
        raise Exception("Invalid acquisition name")


def create_ci_funcs(models, beta):
    """
    Converts GP models into UCB functions and LCB functions.
    :param models: List of N GPflow GPs.
    :param beta: float.
    :return: Tuple, 2 lists of Callables that take in an array of shape (n, N) and return an array of shape (n, 1).
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


def ucb_pne(beta, domain, actions, response_dicts):
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


def ucb_mne(beta, domain, M, rng):
    def acq(models):
        """
        Returns a pair of mixed strategies, and a batch of points to query next. Size of the batch will depend on the
        size of the supports of the potential MNE found.
        :param models: List of N GPflow GPs.
        :return: Tuple (tuple of 2 arrays of shape (M,), array of shape (B, N))
        """
        ucb_funcs, lcb_funcs = create_ci_funcs(models=models, beta=beta)
        U1upper = np.reshape(ucb_funcs[0](domain), (M, M))
        U1lower = np.reshape(lcb_funcs[0](domain), (M, M))
        U2upper = np.reshape(ucb_funcs[1](domain), (M, M))
        U2lower = np.reshape(lcb_funcs[1](domain), (M, M))

        mne = SEM_var_utility(
            U1upper=U1upper,
            U1lower=U1lower,
            U2upper=U2upper,
            U2lower=U2lower,
            num_rand_dists_per_agent=5,
            rng=rng,
            mode="first",
        )[0]

        (s1, s2), (a1supp, a2supp) = get_strategies_and_support(mne, M, M)

        samples_coords = cross_product(a1supp[:, None], a2supp[:, None])
        exploring_samples_coords = []
        for a1 in a1supp:
            a1_ucb = U1upper[a1]
            a1_ucb_argmax_a2coord = np.argmax(a1_ucb)
            if (
                a1_ucb_argmax_a2coord not in a2supp
            ):  # if it is, we would already have sampled this
                exploring_samples_coords.append([a1, a1_ucb_argmax_a2coord])
        for a2 in a2supp:
            a2_ucb = U2upper[:, a2]
            a2_ucb_argmax_a1coord = np.argmax(a2_ucb)
            if (
                a2_ucb_argmax_a1coord not in a1supp
            ):  # if it is, we would already have sampled this
                exploring_samples_coords.append([a2_ucb_argmax_a1coord, a2])
        if len(exploring_samples_coords) != 0:
            all_coords = np.concatenate(
                [samples_coords, np.array(exploring_samples_coords)], axis=0
            )  # (B, 2)
        else:
            all_coords = samples_coords
        all_domain_idxs = all_coords[:, 0] * M + all_coords[:, 1]

        return domain[all_domain_idxs], (s1, s2)

    return acq
