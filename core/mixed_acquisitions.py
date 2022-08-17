import numpy as np

from core.mne import SEM, get_strategies_and_support
from core.utils import cross_product
from core.models import create_ci_funcs


def get_acq_mixed(acq_name, beta, domain, num_actions):
    if acq_name == "ucb_mne":
        return ucb_mne(beta=beta, domain=domain, M=num_actions)
    elif acq_name == "ucb_mne_noexplore":
        return ucb_mne_noexplore(beta=beta, domain=domain, M=num_actions)
    elif acq_name == "max_ent_mne":
        return max_ent_mne(beta=beta, domain=domain, M=num_actions)
    elif acq_name == "random_mne":
        return random_mne(beta=beta, domain=domain, M=num_actions)
    else:
        raise Exception("Invalid acquisition name passed to get_acq_mixed")


def ucb_mne(beta, domain, M):
    def acq(models, prev_successes, rng):
        """
        Algorithm 2. Returns a pair of mixed strategies, the pure strategy to sample, and a list of previous successful
        supports to speed up SEM.
        :param models: List of N GPflow GPs.
        :param prev_successes:
        :return:
        """
        ucb_funcs, lcb_funcs = create_ci_funcs(models=models, beta=beta)
        U1upper = np.reshape(ucb_funcs[0](domain), (M, M))
        U1lower = np.reshape(lcb_funcs[0](domain), (M, M))
        U2upper = np.reshape(ucb_funcs[1](domain), (M, M))
        U2lower = np.reshape(lcb_funcs[1](domain), (M, M))

        # Sample utility functions (line 3 in Algorithm 2)
        U1_sample = rng.uniform(low=U1lower, high=U1upper)
        U2_sample = rng.uniform(low=U2lower, high=U2upper)

        # Report mixed strategy profile (line 4 in Algorithm 2)
        mne_list, prev_successes = SEM(
            U1=U1_sample, U2=U2_sample, mode="first", prev_successes=prev_successes
        )
        mne = mne_list[0]
        (s1, s2), (a1supp, a2supp) = get_strategies_and_support(mne, M, M)

        noreg_samples_coords = cross_product(a1supp[:, None], a2supp[:, None])
        # Compute the exploiting pure strategy profile (Equation 14)
        final_idxs = []
        noreg_idxs = noreg_samples_coords[:, 0] * M + noreg_samples_coords[:, 1]  # (c,)
        noreg_ci_vals = ucb_funcs[0](domain[noreg_idxs]) - lcb_funcs[0](
            domain[noreg_idxs]
        )  # (c, 1)
        noreg_final_idx = noreg_idxs[np.argmax(noreg_ci_vals[:, 0])]
        final_idxs.append(noreg_final_idx)

        # Compute all exploring pure strategy profiles (Equation 12)
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

        # Compute which exploring sample to take
        a1_ucb_br_val = np.max(U1upper @ s2)
        a2_ucb_br_val = np.max(s1 @ U2upper)
        a1_lcb_val = s1 @ U1lower @ s2
        a2_lcb_val = s1 @ U2lower @ s2
        f_check_1 = a1_lcb_val - a1_ucb_br_val
        f_check_2 = a2_lcb_val - a2_ucb_br_val
        if f_check_1 <= f_check_2:
            final_idxs.append(a1_final_idx)
        else:
            final_idxs.append(a2_final_idx)

        pure_strategies = domain[np.array(final_idxs)]

        # Select the strategy with the highest predictive variance to sample (line 5 in Algorithm 2)
        _, variances = models[0].posterior().predict_f(pure_strategies)  # (2, 1)
        sampled_pure_strategy = pure_strategies[np.argmax(np.squeeze(variances))]

        return (s1, s2), sampled_pure_strategy, prev_successes

    return acq


def ucb_mne_noexplore(beta, domain, M):
    """
    UCB-MNE without the exploring samples.
    :param beta:
    :param domain:
    :param M:
    :return:
    """

    def acq(models, prev_successes, rng):
        """
        Returns a pair of mixed strategies, and a batch of points to query next. Size of the batch will depend on the
        size of the supports of the potential MNE found.
        :param models: List of N GPflow GPs.
        :param prev_successes:
        :return:
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
        # Select the strategy with the predictive variance to sample
        noreg_idxs = noreg_samples_coords[:, 0] * M + noreg_samples_coords[:, 1]  # (c,)
        noreg_ci_vals = ucb_funcs[0](domain[noreg_idxs]) - lcb_funcs[0](
            domain[noreg_idxs]
        )  # (c, 1)
        noreg_final_idx = noreg_idxs[np.argmax(noreg_ci_vals[:, 0])]

        sampled_pure_strategy = domain[noreg_final_idx]

        return (s1, s2), sampled_pure_strategy, prev_successes

    return acq


def max_ent_mne(beta, domain, M):
    """
    Pure exploration. Simply chooses the point in domain with highest uncertainty.
    :param beta:
    :param domain:
    :param M:
    :return:
    """

    def acq(models, prev_successes, rng):
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
        (s1, s2), _ = get_strategies_and_support(mne, M, M)

        _, var = (
            models[0].posterior().predict_f(domain)
        )  # use first one because all models have same pred var
        max_ent_idx = np.argmax(np.squeeze(var, axis=-1))
        sampled_pure_strategy = domain[max_ent_idx]

        return (s1, s2), sampled_pure_strategy, prev_successes

    return acq


def random_mne(beta, domain, M):
    """
    Random sampling.
    :param beta:
    :param domain:
    :param M:
    :return:
    """

    def acq(models, prev_successes, rng):
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
        (s1, s2), _ = get_strategies_and_support(mne, M, M)

        rand_idx = rng.integers(0, len(domain))
        sampled_pure_strategy = domain[rand_idx]

        return (s1, s2), sampled_pure_strategy, prev_successes

    return acq
