import numpy as np
from scipy.stats import norm
from statsmodels.sandbox.distributions.extras import mvnormcdf

from core.pne import find_PNE_discrete
from core.mne import SEM, get_strategies_and_support
from core.utils import cross_product, maximize_fn, maxmin_fn
from core.models import create_ci_funcs


def get_acq_pure(
    acq_name,
    beta,
    bounds,
    agent_dims_bounds,
    mode,
    n_samples_outer,
    noise_variance,
    domain=None,
    response_dicts=None,
    num_actions=None,
    inner_max_mode=None
):
    if acq_name == "ucb_pne":
        return ucb_pne(
            beta=beta,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            n_samples_outer=n_samples_outer,
            inner_max_mode=inner_max_mode
        )
    elif acq_name == "prob_eq":
        if domain is None or response_dicts is None or num_actions is None:
            raise Exception("None params passed to prob_eq")
        return prob_eq(
            domain=domain, response_dicts=response_dicts, num_actions=num_actions
        )
    elif acq_name == "SUR":
        if domain is None or response_dicts is None or num_actions is None:
            raise Exception("None params passed to SUR")
        return SUR(
            domain=domain,
            response_dicts=response_dicts,
            num_actions=num_actions,
            noise_variance=noise_variance,
        )
    elif acq_name == "BN":
        return BN(
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            gamma=beta,
        )
    else:
        raise Exception("Invalid acquisition name passed to get_acq_pure")


def get_acq_mixed(acq_name, beta, domain, num_actions):
    if acq_name == "ucb_mne":
        return ucb_mne(beta=beta, domain=domain, M=num_actions)
    elif acq_name == "ucb_mne_noexplore":
        return ucb_mne_noexplore(beta=beta, domain=domain, M=num_actions)
    elif acq_name == "max_ent_mne":
        return max_ent_mne(beta=beta, domain=domain, M=num_actions)
    else:
        raise Exception("Invalid acquisition name passed to get_acq_mixed")


def ucb_pne(beta, bounds, agent_dims_bounds, mode, n_samples_outer, inner_max_mode):
    def acq(models, rng, args_dict):
        """
        Returns 2 points to query next. First one is no-regret selection, second is exploring sample.
        :param models: List of N GPflow GPs.
        :param rng:
        :param args_dict:
        :return: array of shape (2, N).
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
            inner_max_mode=inner_max_mode
        )
        samples.append(noreg_sample)

        # Pick exploring samples
        if mode == "DIRECT":
            exploring_max_mode = "DIRECT"
        else:
            exploring_max_mode = "L-BFGS-B"

        pot_exploring_samples = []
        pot_exploring_vals = []
        noreg_lcb_vals = []
        for i in range(N):
            start_dim, end_dim = agent_dims_bounds[i]
            s_before = noreg_sample[:start_dim]
            s_after = noreg_sample[end_dim:]

            ucb_func = ucb_funcs[i]
            max_ucb_sample, max_ucb_val = maximize_fn(
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
            pot_exploring_samples.append(np.concatenate([s_before, max_ucb_sample, s_after]))
            pot_exploring_vals.append(max_ucb_val)

            noreg_lcb_vals.append(np.squeeze(lcb_funcs[i](noreg_sample[None, :])))

        exploring_scores = np.array(noreg_lcb_vals) - np.array(pot_exploring_vals)
        assert exploring_scores.shape == (N,)
        exploring_sample = pot_exploring_samples[np.argmin(exploring_scores)]
        samples.append(exploring_sample)
        strategies = np.array(samples)  # (2, dims)

        # Select the strategy with the highest predictive variance to sample
        _, variances = models[0].posterior().predict_f(strategies)  # (2, 1)
        sampled_strategy = strategies[np.argmax(np.squeeze(variances))]
        reported_strategy = noreg_sample

        return reported_strategy, sampled_strategy, args_dict

    return acq, {}


def BN(
    bounds,
    agent_dims_bounds,
    mode,
    gamma,
    explore_prob=0.05,
    n_samples_estimate=1000,
):
    """
    Acquisition function from Al-Dujaili et. al. (2018).
    :param bounds:
    :param agent_dims_bounds:
    :param mode:
    :param gamma:
    :param explore_prob:
    :param n_samples_estimate:
    :return:
    """

    def acq(models, rng, args_dict):
        N = len(agent_dims_bounds)

        if mode == "DIRECT":
            max_mode = "DIRECT"
        else:
            max_mode = "L-BFGS-B"

        is_exploring = bool(rng.binomial(n=1, p=explore_prob))
        if not is_exploring:
            # Use BN acquisition function
            def obj(s):
                b = len(s)
                estimated_regs = []
                for i in range(N):
                    start_dim, end_dim = agent_dims_bounds[i]
                    s_before = s[:, :start_dim]  # (b, dims - d_i)
                    s_before = s_before[:, None, :]  # ( b, 1, dims - d_i)
                    s_after = s[:, end_dim:]  # (b, dims - d_i)
                    s_after = s_after[:, None, :]  # ( b, 1, dims - d_i)

                    # Sample n_samples_estimate of points in agent i's action space
                    X_i = rng.uniform(
                        low=bounds[start_dim:end_dim, 0],
                        high=bounds[start_dim:end_dim, 1],
                        size=(n_samples_estimate, end_dim - start_dim),
                    )
                    X_i = X_i[
                        None, :, :
                    ]  # (1, n_samples_estimate, end_dim - start_dim)
                    X = np.concatenate(
                        [
                            np.tile(s_before, (1, n_samples_estimate, 1)),
                            np.tile(X_i, (b, 1, 1)),
                            np.tile(s_after, (1, n_samples_estimate, 1)),
                        ],
                        axis=-1,
                    )

                    X_preds, _ = (
                        models[i].posterior().predict_f(X)
                    )  # (b, n_samples_estimate, 1)
                    assert X_preds.shape == (b, n_samples_estimate, 1)

                    estimated_mean = np.mean(X_preds, axis=1)  # (b, 1)
                    estimated_var = np.mean(
                        (X_preds - estimated_mean[:, None, :]) ** 2, axis=1
                    )  # (b, 1)
                    estimated_std = np.sqrt(estimated_var)  # (b, 1)
                    s_mean, _ = models[i].posterior().predict_f(s)  # (b, 1)

                    estimated_regs.append(
                        np.squeeze(
                            (estimated_mean + gamma * estimated_std - s_mean)
                            / estimated_std,
                            axis=-1,
                        )
                    )
                estimated_regs = np.array(estimated_regs)  # (N, b)
                assert estimated_regs.shape == (N, b)
                return -np.max(estimated_regs, axis=0)[:, None]  # (b, 1)

            ret_sample, _ = maximize_fn(
                f=obj,
                bounds=bounds,
                rng=rng,
                mode=max_mode,
                n_warmup=10,
            )
        else:
            # Pick point with highest uncertainty
            def obj(s):
                b = len(s)
                stds = []
                for i in range(N):
                    _, s_var = models[i].posterior().predict_f(s)  # (b, 1)
                    s_std = np.sqrt(s_var)  # (b, 1)
                    stds.append(np.squeeze(s_std, axis=-1))
                stds = np.array(stds)  # (N, b)
                assert stds.shape == (N, b)
                return np.max(stds, axis=0)[:, None]  # (b, 1)

            ret_sample, _ = maximize_fn(
                f=obj,
                bounds=bounds,
                rng=rng,
                mode=max_mode,
                n_warmup=100,
            )

        return ret_sample[None, :], args_dict

    return acq, {}


def compute_prob_eq_vals(X_idxs, models, domain, num_actions, response_dicts):
    """

    :param X_idxs: domain indices we wish to compute these values on. Array of shape (b, ).
    :param models:
    :param domain:
    :param num_actions:
    :param response_dicts:
    :return:
    """
    print("Computing prob_eq_vals")
    num_agents = len(models)
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

    for s_idx in X_idxs:
        s = domain[s_idx]
        for i in range(num_agents):
            if not is_calculated[s_idx, i]:
                response_idxs = response_dicts[i][s.tobytes()]
                response_points = domain[response_idxs]
                mean, cov = (
                    models[i].posterior().predict_f(response_points, full_cov=True)
                )
                cov = np.squeeze(cov.numpy(), axis=0)
                assert num_actions == len(mean)
                for k, M in enumerate(mats):
                    idx = response_idxs[k]
                    if not is_calculated[idx, i] and idx in X_idxs:
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

    assert is_calculated[X_idxs].all()
    prob_eq_vals = np.prod(probs[X_idxs], axis=-1)  # (b, )

    return prob_eq_vals


def prob_eq(domain, response_dicts, num_actions):
    """
    Probability of Equilibrium acquisition from Picheny et. al. (2018). Requires a discretization of the continuous
    domain, and calculated response dicts.
    :param domain: Array of shape (n, dims).
    :param response_dicts: list of N dicts. Each is a dictionary with keys that are the bytes of a length dims array,
    and returns the idxs of domain that have the actions of all other agents except i the same.
    :param num_actions: int. WARNING: assumes all agents have the same num_actions.
    :return: Array of shape (n,). The probability of equilibrium for each point in domain.
    """
    num_agents = len(response_dicts)
    assert num_actions**num_agents == len(domain)

    def acq(models, rng, args_dict):
        prob_eq_vals = compute_prob_eq_vals(
            X_idxs=np.arange(len(domain)),
            models=models,
            domain=domain,
            num_actions=num_actions,
            response_dicts=response_dicts,
        )
        return domain[np.argmax(prob_eq_vals)][None, :], args_dict

    return acq, {}


def compute_NE_matrix(fvals, domain, response_dicts):
    """

    :param fvals: array of shape (num_draws, n, num_agents).
    :param domain:
    :param response_dicts:
    :return:
    """
    ne_vals = []
    for fval in fvals:
        _, idx = find_PNE_discrete(
            u=fval,
            domain=domain,
            response_dicts=response_dicts,
            is_u_func=False,
        )
        ne_vals.append(fval[idx])
    ne_vals = np.array(ne_vals).T  # (num_agents, num_draws)
    return ne_vals


def estimate_entropy(fvals, domain, response_dicts):
    """
    Gamma-hat from Picheny et. al. (2018), page 8.
    :param fvals: array of shape (num_draws, n, num_agents). Realizations of random functions.
    :param domain:
    :param response_dicts:
    :return: scalar.
    """
    ne_vals = compute_NE_matrix(
        fvals=fvals, domain=domain, response_dicts=response_dicts
    )  # (num_agents, num_draws)
    cov = np.cov(m=ne_vals)
    return np.linalg.det(cov)


def eq_entropy(
    X_idxs,
    models,
    domain,
    response_dicts,
    noise_variance,
    rng,
    num_draws=20,
    num_point_samples=20,
):
    """
    Calculates the NE entropy by conditioning on each point in X. From Picheny et. al. (2018), equation (17).
    Smaller is better, so take the argmin after calculating these values.
    :param X_idxs: domain indices we wish to compute these values on. Array of shape (b, ).
    :param models: List of N GPflow GPs.
    :param domain: Array of shape (n, dims).
    :param response_dicts:
    :param noise_variance:
    :param rng: NumPy rng object.
    :param num_draws: int. Number of GP draws.
    :param num_point_samples: int. Number of samples per point in the domain to condition the GP draws on.
    :return: array of shape (b, ). Entropy of each point in the domain.
    """
    print("Computing entropies")
    X = domain[X_idxs]
    b = len(X_idxs)
    n = len(domain)
    num_agents = len(models)
    # Draw num_draws number of realizations of random functions from GP
    print("eq_ent: drawing num_draws random functions from GP")
    gp_draws = []
    means = []
    covs = []
    for model in models:
        mean, cov = model.posterior().predict_f(domain, full_cov=True)
        mean, cov = np.squeeze(mean, axis=-1), np.squeeze(cov, axis=0)
        gp_draw = rng.multivariate_normal(
            mean=mean, cov=cov, size=num_draws
        )  # slow for large domain
        gp_draws.append(gp_draw)  # (num_draws, n)
        means.append(mean)
        covs.append(cov)

    # For each point in X, draw num_point_samples number of samples. Use faster method of sampling
    # since draws are independent
    print("eq_ent: drawing num_point_samples number of samples")
    sample_draws = []
    X_varis = []
    for i, model in enumerate(models):
        _, var = model.posterior().predict_f(X)  # (b, 1)
        X_varis.append(np.squeeze(var, axis=-1))
        sample = rng.normal(size=(b, num_point_samples))
        sqrt_cov = np.sqrt(np.diag(X_varis[i] + noise_variance))  # (b, b)
        sample_draws_i = (
            sqrt_cov @ sample + means[i][X_idxs, None]
        )  # (b, num_point_samples)
        sample_draws.append(sample_draws_i)

    # For each point in X and for each point sample, condition the random functions on that sample
    print("eq_ent: conditioning random functions on samples using FOXY")
    all_Y_cond_F = []
    for i in range(num_agents):
        lambdas = (
            covs[i][X_idxs] / X_varis[i][:, None]
        )  # (b, n) matrix where each row is a lambda
        assert lambdas.shape == (b, n)

        gp_draw = gp_draws[i]  # (num_draws, n)
        sample_draw = sample_draws[i]  # (b, num_point_samples)

        A = (
            sample_draw.T[:, None, :] - gp_draw[:, X_idxs]
        )  # (num_point_samples, num_draws, b)
        B = A[..., None] * lambdas  # (num_point_samples, num_draws, b, n)
        Y_cond_F = (
            B + gp_draw[None, :, X_idxs, None]
        )  # (num_point_samples, num_draws, b, n)
        all_Y_cond_F.append(Y_cond_F)
    all_Y_cond_F = np.array(
        all_Y_cond_F
    )  # (num_agents, num_point_samples, num_draws, b, n)
    all_Y_cond_F = np.transpose(
        all_Y_cond_F, [3, 1, 2, 4, 0]
    )  # (b, num_point_samples, num_draws, n, num_agents)
    assert all_Y_cond_F.shape == (b, num_point_samples, num_draws, n, num_agents)

    # For each point in domain, estimate entropy
    print("eq_ent: for each point in X, estimate entropy")
    scores = []
    for j in range(b):
        entropies = []
        for k in range(num_point_samples):
            entropy = estimate_entropy(
                fvals=all_Y_cond_F[j, k],
                domain=domain,
                response_dicts=response_dicts,
            )
            entropies.append(entropy)
        scores.append(np.mean(entropies))

    # Prepare gp_draws for next iteration
    gp_draws = np.array(gp_draws)  # (num_agents, num_draws, n)
    assert gp_draws.shape == (num_agents, num_draws, n)
    gp_draws = np.transpose(gp_draws, [1, 2, 0])

    return scores, gp_draws


def C_target(X_idxs, models, domain, response_dicts):
    """
    Computes the target criterion for X.
    :param X_idxs: domain indices we wish to compute these values on. Array of shape (b, ).
    :param models:
    :param domain:
    :param response_dicts:
    :return: Scores. Array of size (b, ).
    """
    print("Computing C_target")
    X = domain[X_idxs]
    post_means = []
    for i, model in enumerate(models):
        mean, _ = model.posterior().predict_f(domain)  # (n, 1)
        post_means.append(np.squeeze(mean, axis=-1))
    post_means = np.array(post_means).T  # (n, num_agents)
    fvals = post_means[None, ...]  # (1, n, num_agents)

    ne_vals = compute_NE_matrix(
        fvals=fvals, domain=domain, response_dicts=response_dicts
    )  # (num_agents, 1)

    means = []
    varis = []
    for i, model in enumerate(models):
        mean, var = model.posterior().predict_f(X)  # (b, 1)
        means.append(np.squeeze(mean, axis=-1))
        varis.append(np.squeeze(var, axis=-1))
    means = np.array(means)  # (num_agents, b)
    varis = np.array(varis)  # (num_agents, b)

    likelihood_per_agent = norm.pdf(
        (ne_vals - means) / np.sqrt(varis)
    )  # (num_agents, b)
    likelihood = np.prod(likelihood_per_agent, axis=0)  # (b, )
    return likelihood


def C_box(X_idxs, fvals, models, domain, response_dicts):
    """
    Computes the box criterion for X.
    :param X_idxs: domain indices we wish to compute these values on. Array of shape (b, ).
    :param fvals: array of shape (num_draws, n, num_agents). Realizations of random functions.
    :param models:
    :param domain:
    :param response_dicts:
    :return: Scores. Array of size (b, ).
    """
    print("Computing C_box")
    X = domain[X_idxs]
    ne_vals = compute_NE_matrix(
        fvals=fvals, domain=domain, response_dicts=response_dicts
    )  # (num_agents, num_draws)
    T_Li = np.min(ne_vals, axis=-1)  # (num_agents)
    T_Ui = np.max(ne_vals, axis=-1)  # (num_agents)

    means = []
    varis = []
    for i, model in enumerate(models):
        mean, var = model.posterior().predict_f(X)  # (b, 1)
        means.append(np.squeeze(mean, axis=-1))
        varis.append(np.squeeze(var, axis=-1))
    means = np.array(means)  # (num_agents, b)
    varis = np.array(varis)  # (num_agents, b)

    prob_less_upper = norm.cdf(
        (T_Ui[:, None] - means) / np.sqrt(varis)
    )  # (num_agents, b)
    prob_less_lower = norm.cdf(
        (T_Li[:, None] - means) / np.sqrt(varis)
    )  # (num_agents, b)
    prob_box = np.prod(prob_less_upper - prob_less_lower, axis=0)  # (b, )

    return prob_box


def SUR(
    domain, response_dicts, num_actions, noise_variance, num_sim=1296, num_cand=256
):
    if num_sim > len(domain):
        num_sim = len(domain)
    if num_cand > num_sim:
        num_cand = num_sim

    args_dict = {"prev_gp_draws": None}

    def acq(models, rng, args_dict):
        prev_gp_draws = args_dict["prev_gp_draws"]
        if prev_gp_draws is None:
            sim_scores = C_target(
                X_idxs=np.arange(len(domain)),
                models=models,
                domain=domain,
                response_dicts=response_dicts,
            )
        else:
            sim_scores = C_box(
                X_idxs=np.arange(len(domain)),
                fvals=prev_gp_draws,
                models=models,
                domain=domain,
                response_dicts=response_dicts,
            )
        assert len(sim_scores) == len(domain)

        X_sim_idxs = np.argpartition(sim_scores, -num_sim)[-num_sim:]

        cand_scores = compute_prob_eq_vals(
            X_idxs=X_sim_idxs,
            models=models,
            domain=domain,
            num_actions=num_actions,
            response_dicts=response_dicts,
        )
        assert len(cand_scores) == len(X_sim_idxs)

        # these idxs are with respect to cand_scores
        X_cand_idxs_inter = np.argpartition(cand_scores, -num_cand)[-num_cand:]
        X_cand_idxs = X_sim_idxs[
            X_cand_idxs_inter
        ]  # these are now with respect to domain

        entropy_scores, gp_draws = eq_entropy(
            X_idxs=X_cand_idxs,
            models=models,
            domain=domain,
            response_dicts=response_dicts,
            noise_variance=noise_variance,
            rng=rng,
        )
        assert len(entropy_scores) == len(X_cand_idxs)
        # this idx is with respect to entropy_scores
        args_dict["prev_gp_draws"] = gp_draws

        return domain[X_cand_idxs[np.argmin(entropy_scores)]][None, :], args_dict

    return acq, args_dict


def ucb_mne(beta, domain, M):
    def acq(models, prev_successes, rng):
        """
        Returns a pair of mixed strategies, and a batch of points to query next. One no-regret pure strategy, and one
        exploring pure strategy.
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

        # Select the strategy with the highest predictive variance to sample
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
        # Take the noreg sample with the highest uncertainty
        final_idxs = []
        noreg_idxs = noreg_samples_coords[:, 0] * M + noreg_samples_coords[:, 1]  # (c,)
        noreg_ci_vals = ucb_funcs[0](domain[noreg_idxs]) - lcb_funcs[0](
            domain[noreg_idxs]
        )  # (c, 1)
        noreg_final_idx = noreg_idxs[np.argmax(noreg_ci_vals[:, 0])]
        final_idxs.append(noreg_final_idx)

        samples = domain[np.array(final_idxs)]

        return samples, (s1, s2), prev_successes

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
        samples = domain[max_ent_idx : max_ent_idx + 1]

        return samples, (s1, s2), prev_successes

    return acq
