import numpy as np
from statsmodels.sandbox.distributions.extras import mvnormcdf

from core.utils import (
    maximize_fn,
    maxmin_fn,
    discretize_domain,
    cross_product,
    create_response_dict,
)
from core.models import create_ci_funcs, create_mean_funcs


def get_acq_pure(
    acq_name,
    beta,
    bounds,
    agent_dims_bounds,
    mode,
    n_samples_outer,
    inner_max_mode,
    num_actions=None,
    agent_dims=None,
):
    if acq_name == "ucb_pne":
        return ucb_pne(
            beta=beta,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            n_samples_outer=n_samples_outer,
            inner_max_mode=inner_max_mode,
        )
    elif acq_name == "ucb_pne_noexplore":
        return ucb_pne_noexplore(
            beta=beta,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            n_samples_outer=n_samples_outer,
            inner_max_mode=inner_max_mode,
        )
    elif acq_name == "prob_eq":
        if num_actions is None or agent_dims is None:
            raise Exception("None params passed to prob_eq")
        return prob_eq(
            num_actions=num_actions,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            n_samples_outer=n_samples_outer,
            inner_max_mode=inner_max_mode,
            agent_dims=agent_dims,
        )
    elif acq_name == "BN":
        return BN(
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            gamma=beta,
            n_samples_outer=n_samples_outer,
            inner_max_mode=inner_max_mode,
        )
    else:
        raise Exception("Invalid acquisition name passed to get_acq_pure")


def ucb_pne(beta, bounds, agent_dims_bounds, mode, n_samples_outer, inner_max_mode):
    def acq(models, rng, args_dict):
        """
        Algorithm 1. Returns reported strategy profile, and either the reported or exploring strategy profile to be
        sampled.
        :param models: List of N GPflow GPs.
        :param rng:
        :param args_dict:
        :return:
        """
        N = len(agent_dims_bounds)
        ucb_funcs, lcb_funcs = create_ci_funcs(models=models, beta=beta)
        samples = []
        # Compute reported strategy profile (line 3 of Algorithm 1)
        noreg_sample, _ = maxmin_fn(
            outer_funcs=ucb_funcs,
            inner_funcs=lcb_funcs,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            rng=rng,
            n_samples_outer=n_samples_outer,
            inner_max_mode=inner_max_mode,
        )
        samples.append(noreg_sample)

        # Compute exploring strategy profile (Equation 5)
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
            pot_exploring_samples.append(
                np.concatenate([s_before, max_ucb_sample, s_after])
            )
            pot_exploring_vals.append(max_ucb_val)

            noreg_lcb_vals.append(np.squeeze(lcb_funcs[i](noreg_sample[None, :])))

        exploring_scores = np.array(noreg_lcb_vals) - np.array(pot_exploring_vals)
        assert exploring_scores.shape == (N,)
        exploring_sample = pot_exploring_samples[np.argmin(exploring_scores)]
        samples.append(exploring_sample)
        strategies = np.array(samples)  # (2, dims)

        # Select the strategy with the highest predictive variance to sample (line 4 of Algorithm 1)
        _, variances = models[0].posterior().predict_f(strategies)  # (2, 1)
        sampled_strategy = strategies[np.argmax(np.squeeze(variances))]
        reported_strategy = noreg_sample

        return reported_strategy, sampled_strategy, args_dict

    return acq, {}


def ucb_pne_noexplore(
    beta, bounds, agent_dims_bounds, mode, n_samples_outer, inner_max_mode
):
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
        # Compute reported strategy profile (line 3 of Algorithm 1)
        noreg_sample, _ = maxmin_fn(
            outer_funcs=ucb_funcs,
            inner_funcs=lcb_funcs,
            bounds=bounds,
            agent_dims_bounds=agent_dims_bounds,
            mode=mode,
            rng=rng,
            n_samples_outer=n_samples_outer,
            inner_max_mode=inner_max_mode,
        )
        reported_strategy = noreg_sample
        sampled_strategy = noreg_sample.copy()

        return reported_strategy, sampled_strategy, args_dict

    return acq, {}


def BN(
    bounds,
    agent_dims_bounds,
    mode,
    gamma,
    n_samples_outer,
    inner_max_mode,
    explore_prob=0.05,
    n_samples_estimate=1000,
):
    """
    Acquisition function from Al-Dujaili et. al. (2018).
    :param bounds:
    :param agent_dims_bounds:
    :param mode:
    :param gamma:
    :param n_samples_outer:
    :param inner_max_mode:
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

            sampled_strategy, _ = maximize_fn(
                f=obj,
                bounds=bounds,
                rng=rng,
                mode=max_mode,
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

            sampled_strategy, _ = maximize_fn(
                f=obj,
                bounds=bounds,
                rng=rng,
                mode=max_mode,
            )

        # Report NE computed using predictive mean
        if args_dict["is_reporting"]:
            mean_funcs = create_mean_funcs(models=models)
            reported_strategy, _ = maxmin_fn(
                outer_funcs=mean_funcs,
                inner_funcs=mean_funcs,
                bounds=bounds,
                agent_dims_bounds=agent_dims_bounds,
                mode=mode,
                rng=rng,
                n_samples_outer=n_samples_outer,
                inner_max_mode=inner_max_mode,
            )
        else:
            print("Not reporting")
            reported_strategy = None

        return reported_strategy, sampled_strategy, args_dict

    return acq, {"is_reporting": True}


def compute_prob_eq_vals(X_idxs, models, domain, num_actions, response_dicts):
    """
    Computes probability of equilibrium for points determined by X_idxs.
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
                            maxpts=2000 * num_actions,
                            abseps=1,
                        )
                        probs[idx, i] = prob
                        is_calculated[idx, i] = True

    assert is_calculated[X_idxs].all()
    prob_eq_vals = np.prod(probs[X_idxs], axis=-1)  # (b, )

    return prob_eq_vals


def prob_eq(
    num_actions,
    bounds,
    agent_dims_bounds,
    mode,
    n_samples_outer,
    inner_max_mode,
    agent_dims,
):
    """
    Probability of Equilibrium acquisition from Picheny et. al. (2018). Requires a discretization of the continuous
    domain, and calculated response dicts.
    :param num_actions: int. WARNING: assumes all agents have the same num_actions.
    :param bounds:
    :param agent_dims_bounds:
    :param mode:
    :param n_samples_outer:
    :param inner_max_mode:
    :param agent_dims:
    :return: Array of shape (n,). The probability of equilibrium for each point in domain.
    """
    num_agents = len(agent_dims)

    def acq(models, rng, args_dict):
        # Sample new discrete domain
        domain = discretize_domain(
            num_agents=num_agents,
            num_actions=num_actions,
            bounds=bounds,
            agent_dims=agent_dims,
            rng=rng,
            mode="random",
        )
        domain.flags.writeable = False
        # Create response_dicts
        print("Creating response dicts")
        action_idxs = np.arange(num_actions)
        domain_in_idxs = action_idxs[:, None]
        for i in range(1, num_agents):
            domain_in_idxs = cross_product(domain_in_idxs, action_idxs[:, None])
        response_dicts = [
            create_response_dict(i, domain, domain_in_idxs, action_idxs)
            for i in range(num_agents)
        ]

        prob_eq_vals = compute_prob_eq_vals(
            X_idxs=np.arange(len(domain)),
            models=models,
            domain=domain,
            num_actions=num_actions,
            response_dicts=response_dicts,
        )
        sampled_strategy = domain[np.argmax(prob_eq_vals)]

        if args_dict["is_reporting"]:
            # Report NE computed using predictive mean
            mean_funcs = create_mean_funcs(models=models)
            reported_strategy, _ = maxmin_fn(
                outer_funcs=mean_funcs,
                inner_funcs=mean_funcs,
                bounds=bounds,
                agent_dims_bounds=agent_dims_bounds,
                mode=mode,
                rng=rng,
                n_samples_outer=n_samples_outer,
                inner_max_mode=inner_max_mode,
            )
        else:
            print("Not reporting")
            reported_strategy = None

        return reported_strategy, sampled_strategy, args_dict

    return acq, {"is_reporting": True}
