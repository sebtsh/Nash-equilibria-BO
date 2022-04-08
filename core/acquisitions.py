import numpy as np

from core.pne import best_response_payoff_pure, ucb_f


def get_acquisition(acq_name,
                    beta,
                    domain,
                    actions,
                    response_dicts):
    if acq_name == 'ucb_pne_naive':
        return ucb_pne_naive(beta=beta,
                             domain=domain,
                             actions=actions,
                             response_dicts=response_dicts)
    elif acq_name == 'ucb_pne':
        return ucb_pne(beta=beta,
                       domain=domain,
                       actions=actions,
                       response_dicts=response_dicts)
    else:
        raise Exception("Invalid acquisition name")


def create_ci_funcs(models,
                    beta):
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

    return ([create_ci_func(models[i], is_ucb=True) for i in range(N)],
            [create_ci_func(models[i], is_ucb=False) for i in range(N)])


def ucb_pne_naive(beta,
                  domain,
                  actions,
                  response_dicts):
    def acq(models):
        """
        Returns a point to query next.
        :param models: List of N GPflow GPs.
        :return: array of shape (1, N).
        """
        ucb_funcs, _ = create_ci_funcs(models=models,
                                       beta=beta)
        ucb_brp = best_response_payoff_pure(u=ucb_funcs,
                                            S=domain,
                                            actions=actions,
                                            response_dicts=response_dicts)  # array of shape (M ** N, N)

        next_idx = np.argmin(np.max(ucb_brp, axis=-1))
        return domain[next_idx:next_idx + 1]

    return acq


def ucb_pne(beta,
            domain,
            actions,
            response_dicts):
    def acq(models):
        """
        Returns N + 1 points to query next. First one is no-regret selection, next N are exploring samples.
        :param models: List of N GPflow GPs.
        :return: array of shape (N + 1, N).
        """
        M = len(actions)
        _, N = domain.shape
        ucb_funcs, lcb_funcs = create_ci_funcs(models=models,
                                               beta=beta)
        all_ucb = np.zeros((M ** N, N))
        all_lcb = np.zeros((M ** N, N))
        for j in range(N):
            all_ucb[:, j] = np.squeeze(ucb_funcs[j](domain), axis=-1)
            all_lcb[:, j] = np.squeeze(lcb_funcs[j](domain), axis=-1)

        samples_idxs = []
        # Pick no-regret selection
        ucb_f_vals = ucb_f(all_ucb=all_ucb,
                           all_lcb=all_lcb,
                           S=domain,
                           actions=actions,
                           response_dicts=response_dicts)
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
