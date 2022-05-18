import numpy as np
from statsmodels.sandbox.distributions.extras import mvnormcdf

from core.pne import ucb_f
from core.mne import SEM, get_strategies_and_support
from core.utils import cross_product, maximize_fn, maxmin_fn


def get_acq_pure(
    acq_name,
    beta,
    bounds,
    agent_dims_bounds,
    mode,
    n_samples_outer,
):
    if acq_name == "ucb_pne":
        return ucb_pne(
            beta=beta,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            n_samples_outer=n_samples_outer,
        )
    else:
        raise Exception("Invalid acquisition name")


def get_acq_mixed(acq_name, beta, domain, num_actions):
    if acq_name == "ucb_mne":
        return ucb_mne(beta=beta, domain=domain, M=num_actions)
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
            mean, var = model.posterior().predict_f(X)
            if is_ucb:
                return mean + beta * np.sqrt(var)
            else:  # is lcb
                return mean - beta * np.sqrt(var)

        return inn

    return (
        [create_ci_func(models[i], is_ucb=True) for i in range(N)],
        [create_ci_func(models[i], is_ucb=False) for i in range(N)],
    )


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
        if mode == "DIRECT":
            exploring_max_mode = "DIRECT"
        else:
            exploring_max_mode = "L-BFGS-B"
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
                mode=exploring_max_mode,
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

        U1_sample = rng.uniform(low=U1lower, high=U1upper)
        U2_sample = rng.uniform(low=U2lower, high=U2upper)

        mne_list, prev_successes = SEM(
            U1=U1_sample, U2=U2_sample, mode="first", prev_successes=prev_successes
        )
        mne = mne_list[0]
        (s1, s2), (a1supp, a2supp) = get_strategies_and_support(mne, M, M)

        noreg_samples_coords = cross_product(a1supp[:, None], a2supp[:, None])
        # Take the noreg sample with the highest uncertainty
        final_idxs = []
        noreg_idxs = noreg_samples_coords[:, 0] * M + noreg_samples_coords[:, 1]  # (c,)
        noreg_ci_vals = ucb_funcs[0](domain[noreg_idxs]) - lcb_funcs[0](
            domain[noreg_idxs]
        )  # (c, 1)
        noreg_final_idx = noreg_idxs[np.argmax(noreg_ci_vals[:, 0])]
        final_idxs.append(noreg_final_idx)

        # Exploring samples
        a1_ucb_br_coord = np.argmax(U1upper @ s2)
        a2_ucb_br_coord = np.argmax(s1 @ U2upper)
        a1_exp_coords = cross_product(np.array([[a1_ucb_br_coord]]), a2supp[:, None])
        a2_exp_coords = cross_product(a1supp[:, None], np.array([[a2_ucb_br_coord]]))
        a1_exp_idxs = a1_exp_coords[:, 0] * M + a1_exp_coords[:, 1]  # (c,)
        a2_exp_idxs = a2_exp_coords[:, 0] * M + a2_exp_coords[:, 1]  # (c,)
        a1_exp_ci_vals = ucb_funcs[0](domain[a1_exp_idxs]) - lcb_funcs[0](
            domain[a1_exp_idxs]
        )  # (c, 1)
        a2_exp_ci_vals = ucb_funcs[0](domain[a2_exp_idxs]) - lcb_funcs[0](
            domain[a2_exp_idxs]
        )  # (c, 1)
        a1_final_idx = a1_exp_idxs[np.argmax(a1_exp_ci_vals[:, 0])]
        a2_final_idx = a2_exp_idxs[np.argmax(a2_exp_ci_vals[:, 0])]
        final_idxs.append(a1_final_idx)
        final_idxs.append(a2_final_idx)

        samples = domain[np.array(final_idxs)]
        # print(f"Samples: {samples}")

        return samples, (s1, s2), prev_successes

    return acq


def prob_eq(domain, response_dicts, num_actions):
    """
    Calculates the probability of equilibrium from Picheny et. al. (2018). Requires a discretization of the continuous
    domain, and calculated response dicts.
    :param models:
    :param domain: Array of shape (n, dims).
    :param response_dicts: list of N dicts. Each is a dictionary with keys that are the bytes of a length dims array,
    and returns the idxs of domain that have the actions of all other agents except i the same.
    :param num_actions: int. WARNING: assumes all agents have the same num_actions.
    :return: Array of shape (n,). The probability of equilibrium for each point in domain.
    """
    num_agents = len(response_dicts)
    assert num_actions**num_agents == len(domain)

    def acq(models, rng):
        probs = np.zeros((len(domain), num_agents))
        is_calculated = np.zeros((len(domain), num_agents), dtype=np.bool)
        # Precompute selection matrices
        mats = []
        for k in range(num_actions):
            mat = np.eye(num_actions)
            mat[:, k] = -1
            mat = np.delete(mat, k, axis=0)  # (num_actions - 1, num_actions)
            mats.append(mat)
        mats = np.array(mats)

        for j, s in enumerate(domain):
            for i in range(num_agents):
                if not is_calculated[j, i]:
                    response_idxs = response_dicts[i][s.tobytes()]
                    response_points = domain[response_idxs]
                    mean, cov = (
                        models[i].posterior().predict_f(response_points, full_cov=True)
                    )
                    cov = np.squeeze(cov.numpy(), axis=0)
                    assert num_actions == len(mean)
                    for k, M in enumerate(mats):
                        idx = response_idxs[k]
                        if not is_calculated[idx, i]:
                            mean_reduced = np.squeeze(M @ mean, axis=-1)
                            cov_reduced = M @ cov @ M.T
                            prob = mvnormcdf(
                                upper=np.zeros(num_actions - 1),
                                mu=mean_reduced,
                                cov=cov_reduced,
                                maxpts=5000,
                                abseps=1,
                            )
                            probs[idx, i] = prob
                            is_calculated[idx, i] = True

        assert is_calculated.all()
        prob_eq_vals = np.prod(probs, axis=-1)
        return domain[np.argmax(prob_eq_vals)][None, :]

    return acq


def SUR(models, domain, response_dicts):
    pass
