import itertools
import numpy as np
from scipy.optimize import minimize

from core.utils import unif_in_simplex, sort_size_balance


def get_strategies_and_support(x, M1, M2):
    """
    Takes the output of SEM_var_utility, an MNE represented as an array, and extracts the strategies and support of the
    two agents.
    :param x: array of length M1 + M2 + 2 + 2 * (M1 * M2)
    :param M1: Agent 1 number of actions.
    :param M2: Agent 2 number of actions.
    :return: Tuple (agent 1 strategy, agent 2 strategy), Tuple (agent 1 support, agent 2 support)
    """
    s1 = x[:M1]
    s2 = x[M1 + 1 : M1 + M2 + 1]
    a1support = np.where(np.isclose(np.zeros(M1), s1) == False)[0]
    a2support = np.where(np.isclose(np.zeros(M2), s2) == False)[0]

    return (s1, s2), (a1support, a2support)


def neg_brp_mixed(U1, U2, s):
    """
    Computes the negative best response payoff for each agent for a given mixed strategy profile.
    :param U1: Array of shape (M1, M2). Agent 1's utility matrix. Agent 1 row player, agent 2 column player.
    :param U2: Array of shape (M1, M2). Agent 2's utility matrix. Agent 1 row player, agent 2 column player.
    :param s: Tuple (agent 1 strategy, agent 2 strategy). Each strategy is an array of length M1 or M2.
    :return: Tuple (agent 1 -brp, agent 2 -brp).
    """
    a1_expected_utility = s[0] @ U1 @ s[1]
    a1_best_response = np.max(U1 @ s[1])
    a2_expected_utility = s[0] @ U2 @ s[1]
    a2_best_response = np.max(s[0] @ U2)
    return (
        a1_expected_utility - a1_best_response,
        a2_expected_utility - a2_best_response,
    )


def evaluate_support(
    supports,
    U1upper,
    U1lower,
    U2upper,
    U2lower,
    num_rand_dists_per_agent,
    rng,
    mode,
    num_samples,
):
    s1, s2 = supports
    if mode == "nonlinear":
        cons, bounds = build_tgs_constraints_var_utility(
            s1=s1,
            s2=s2,
            U1upper=U1upper,
            U1lower=U1lower,
            U2upper=U2upper,
            U2lower=U2lower,
        )

        init_points = build_init_points(
            s1=s1,
            s2=s2,
            U1upper=U1upper,
            U1lower=U1lower,
            U2upper=U2upper,
            U2lower=U2lower,
            num_rand_dists_per_agent=num_rand_dists_per_agent,
            rng=rng,
        )

        success, res = tgs_var_utility(
            init_points=init_points, constraints=cons, bounds=bounds
        )
    elif mode == "linear_with_sampling":
        for i in range(num_samples):
            U1 = rng.uniform(low=U1lower, high=U1upper, size=U1lower.shape)
            U2 = rng.uniform(low=U2lower, high=U2upper, size=U2lower.shape)
            cons, bounds = build_tgs_constraints(s1=s1, s2=s2, U1=U1, U2=U2)
            success, res = tgs(constraints=cons, bounds=bounds)
            if success:
                res = np.concatenate([res, U1.ravel(), U2.T.ravel()])
                break
    else:
        raise Exception("Invalid mode passed to evaluate_support")

    return success, res


def SEM_var_utility(
    U1upper,
    U1lower,
    U2upper,
    U2lower,
    num_rand_dists_per_agent,
    rng,
    mode,
    prev_successes,
    evaluation_mode,
    num_samples,
):
    """
    Runs the support-enumeration method (SEM) to find all mixed NEs.
    :param U1upper: Array of shape (M1, M2). Agent 1's upper bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U1lower: Array of shape (M1, M2). Agent 1's lower bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U2upper: Array of shape (M1, M2). Agent 2's upper bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U2lower: Array of shape (M1, M2). Agent 1's lower bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param num_rand_dists_per_agent: int.
    :param rng: NumPy rng object.
    :param mode: str. Either 'all' or 'first'.
    :param prev_successes: list of tuples. Each tuple is a pair of support arrays e.g. ([1, 2], [0, 5]).
    :return: list of all MNEs.
    """
    M1, M2 = U1upper.shape
    pairs = [(x, y) for x in range(1, M1 + 1) for y in range(1, M2 + 1)]
    strategy_pair_order = sort_size_balance(pairs)
    mnes = []
    is_found = False

    # First try the support pairs that gave a valid MNE in previous iterations
    if len(prev_successes) > 0:
        for i, (s1, s2) in enumerate(prev_successes):
            if (
                len(
                    conditionally_dominated_var_utility(
                        a1actions=s1,
                        a2actions=s2,
                        active_agent=1,
                        U1upper=U1upper,
                        U1lower=U1lower,
                        U2upper=U2upper,
                        U2lower=U2lower,
                    )
                )
                == 0
                and len(
                    conditionally_dominated_var_utility(
                        a1actions=s1,
                        a2actions=s2,
                        active_agent=2,
                        U1upper=U1upper,
                        U1lower=U1lower,
                        U2upper=U2upper,
                        U2lower=U2lower,
                    )
                )
                == 0
            ):
                success, res = evaluate_support(
                    supports=(s1, s2),
                    U1upper=U1upper,
                    U1lower=U1lower,
                    U2upper=U2upper,
                    U2lower=U2lower,
                    num_rand_dists_per_agent=num_rand_dists_per_agent,
                    rng=rng,
                    mode=evaluation_mode,
                    num_samples=num_samples,
                )

                if success:
                    print(f"tgs succeeded for s1:{s1} s2:{s2}")
                    mnes.append(res)
                    prev_successes.pop(i)
                    prev_successes.insert(0, (s1, s2))  # Move success to top of list
                    if mode == "first":
                        is_found = True
                        break

                else:
                    print(f"tgs failed for s1:{s1} s2:{s2}")
            else:
                print(f"prev_success {(s1, s2)} conditionally dominated")

    # Now iterate through all possibilities
    if not is_found or mode == "all":
        for pair in strategy_pair_order:
            s1_size, s2_size = pair
            all_s1 = itertools.combinations(np.arange(M1), s1_size)
            for s1 in all_s1:
                s1 = np.array(s1)
                A2 = np.setdiff1d(
                    np.arange(M2),
                    conditionally_dominated_var_utility(
                        a1actions=s1,
                        a2actions=np.arange(M2),
                        active_agent=2,
                        U1upper=U1upper,
                        U1lower=U1lower,
                        U2upper=U2upper,
                        U2lower=U2lower,
                    ),
                )
                if (
                    len(
                        conditionally_dominated_var_utility(
                            a1actions=s1,
                            a2actions=A2,
                            active_agent=1,
                            U1upper=U1upper,
                            U1lower=U1lower,
                            U2upper=U2upper,
                            U2lower=U2lower,
                        )
                    )
                    == 0
                ):
                    all_s2 = itertools.combinations(A2, s2_size)
                    for s2 in all_s2:
                        s2 = np.array(s2)
                        if (
                            len(
                                conditionally_dominated_var_utility(
                                    a1actions=s1,
                                    a2actions=s2,
                                    active_agent=1,
                                    U1upper=U1upper,
                                    U1lower=U1lower,
                                    U2upper=U2upper,
                                    U2lower=U2lower,
                                )
                            )
                            == 0
                        ):  # if true, run TGS
                            success, res = evaluate_support(
                                supports=(s1, s2),
                                U1upper=U1upper,
                                U1lower=U1lower,
                                U2upper=U2upper,
                                U2lower=U2lower,
                                num_rand_dists_per_agent=num_rand_dists_per_agent,
                                rng=rng,
                                mode=evaluation_mode,
                                num_samples=num_samples,
                            )
                            if success:
                                print(f"tgs succeeded for s1:{s1} s2:{s2}")
                                # print("U1upper:")
                                # print(U1upper)
                                # print("U1lower:")
                                # print(U1lower)
                                # print("U2upper:")
                                # print(U2upper)
                                # print("U2lower:")
                                # print(U2lower)
                                mnes.append(res)
                                prev_successes.insert(
                                    0, (s1, s2)
                                )  # Move success to top of list
                                if mode == "first":
                                    is_found = True
                                    break
                            else:
                                print(f"tgs failed for s1:{s1} s2:{s2}")
                if mode == "first" and is_found:
                    break
            if mode == "first" and is_found:
                break

    if len(mnes) == 0:
        print("SEM_var_utility failed")
        # print(f"U1upper:{U1upper}")
        # print(f"U1lower:{U1lower}")
        # print(f"U2upper:{U2upper}")
        # print(f"U2lower:{U2lower}")
        # pickle.dump((U1upper, U1lower, U2upper, U2lower), open("results/mne/U1U2bounds", "wb"))

    return mnes, prev_successes


def conditionally_dominated_var_utility(
    a1actions, a2actions, active_agent, U1upper, U1lower, U2upper, U2lower
):
    """
    Determines which actions of the active player are conditionally dominated given the action set of the other player.
    Takes an optimistic approach in which an action is only conditionally dominated if another of that player's action's
    lower bounds are all higher than the first action's upper bounds.
    :param a1actions: 1-D array.
    :param a2actions: 1-D array.
    :param active_agent: Either 1 or 2. Indicates the agent whose actions will be determined to be dominated
    conditioned on the other agent's actions.
    :param U1upper: Array of shape (M1, M2). Agent 1's upper bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U1lower: Array of shape (M1, M2). Agent 1's lower bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U2upper: Array of shape (M1, M2). Agent 2's upper bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U2lower: Array of shape (M1, M2). Agent 1's lower bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :return: 1-D array of indices of actions of the active player that are conditionally dominated.
    """
    if active_agent == 1:
        U1_cond = U1upper[:, a2actions]  # (M1, |s2|)
        diff_mat = U1_cond[:, None, :] - U1lower[:, a2actions]  # (M1, M1, |s2|)
        diff_mat_max = np.max(diff_mat, axis=-1)  # (M1, M1)
        return np.intersect1d(
            a1actions, np.array(list(set(np.where(diff_mat_max < 0)[0])))
        )
    elif active_agent == 2:
        U2_cond = U2upper[a1actions, :].T  # (M2, |s1|)
        diff_mat = U2_cond[:, None, :] - U2lower[a1actions, :].T  # (M2, M2, |s1|)
        diff_mat_max = np.max(diff_mat, axis=-1)  # (M2, M2)
        return np.intersect1d(
            a2actions, np.array(list(set(np.where(diff_mat_max < 0)[0])))
        )
    else:
        raise NotImplemented


def SEM(U1, U2, mode, prev_successes):
    """
    Runs the support-enumeration method (SEM) to find all mixed NEs.
    :param U1: Array of shape (M1, M2). Agent 1's utility matrix. Agent 1 row player, agent 2 column player.
    :param U2: Array of shape (M1, M2). Agent 2's utility matrix. Agent 1 row player, agent 2 column player.
    :param mode: str. Either "all" or "first".
    :param prev_successes: list of tuples. Each tuple is a pair of support arrays e.g. ([1, 2], [0, 5]).
    :return: list of all MNEs.
    """
    M1, M2 = U1.shape
    is_found = False
    mnes = []
    # If we only want one, first try previous successes
    if mode == "first" and len(prev_successes) > 0:
        for i, (s1, s2) in enumerate(prev_successes):
            if (
                len(
                    conditionally_dominated(
                        a1actions=s1,
                        a2actions=s2,
                        active_agent=1,
                        U1=U1,
                        U2=U2,
                    )
                )
                == 0
                and len(
                    conditionally_dominated(
                        a1actions=s1,
                        a2actions=s2,
                        active_agent=2,
                        U1=U1,
                        U2=U2,
                    )
                )
                == 0
            ):
                cons, bounds = build_tgs_constraints(s1=s1, s2=s2, U1=U1, U2=U2)
                success, res = tgs(constraints=cons, bounds=bounds)
                if success:
                    # print("Using previous success!")
                    mnes.append(res)
                    prev_successes.pop(i)
                    prev_successes.insert(0, (s1, s2))  # Move success to top of list
                    if mode == "first":
                        is_found = True
                        break

    if not is_found:
        pairs = [(x, y) for x in range(1, M1 + 1) for y in range(1, M2 + 1)]
        sorted_pairs = sort_size_balance(pairs)
        for pair in sorted_pairs:
            s1_size, s2_size = pair
            all_s1 = itertools.combinations(np.arange(M1), s1_size)
            for s1 in all_s1:
                s1 = np.array(s1)
                A2 = np.setdiff1d(
                    np.arange(M2),
                    conditionally_dominated(
                        a1actions=s1,
                        a2actions=np.arange(M2),
                        active_agent=2,
                        U1=U1,
                        U2=U2,
                    ),
                )
                if (
                    len(
                        conditionally_dominated(
                            a1actions=s1, a2actions=A2, active_agent=1, U1=U1, U2=U2
                        )
                    )
                    == 0
                ):
                    all_s2 = itertools.combinations(A2, s2_size)
                    for s2 in all_s2:
                        s2 = np.array(s2)
                        if (
                            len(
                                conditionally_dominated(
                                    a1actions=s1,
                                    a2actions=s2,
                                    active_agent=1,
                                    U1=U1,
                                    U2=U2,
                                )
                            )
                            == 0
                        ):  # if true, run TGS
                            cons, bounds = build_tgs_constraints(
                                s1=s1, s2=s2, U1=U1, U2=U2
                            )
                            success, res = tgs(constraints=cons, bounds=bounds)
                            if success:
                                mnes.append(res)
                                prev_successes.insert(
                                    0, (s1, s2)
                                )  # Move success to top of list
                                if mode == "first":
                                    is_found = True
                                    break
                if mode == "first" and is_found:
                    break
            if mode == "first" and is_found:
                break
    return mnes, prev_successes  # no prev_successes optimization for now


def conditionally_dominated(a1actions, a2actions, active_agent, U1, U2):
    """
    Determines which actions of the active player are conditionally dominated given the action set of the other player.
    :param a1actions: 1-D array.
    :param a2actions: 1-D array.
    :param active_agent: Either 1 or 2. Indicates the player whose actions will be determined to be dominated
    conditioned on the other player's actions.
    :param U1: Array of shape (M1, M2). Agent 1's utility matrix. Agent 1 row player, agent 2 column player.
    :param U2: Array of shape (M1, M2). Agent 2's utility matrix. Agent 1 row player, agent 2 column player.
    :return: 1-D array of indices of actions of the active player that are conditionally dominated.
    """
    if active_agent == 1:
        U1_cond = U1[:, a2actions]  # (M1, |s2|)
        diff_mat = U1_cond[:, None, :] - U1_cond  # (M1, M1, |s2|)
        diff_mat_max = np.max(diff_mat, axis=-1)  # (M1, M1)
        return np.intersect1d(
            a1actions, np.array(list(set(np.where(diff_mat_max < 0)[0])))
        )
    elif active_agent == 2:
        U2_cond = U2[a1actions].T  # (M2, |s1|)
        diff_mat = U2_cond[:, None, :] - U2_cond  # (M2, M2, |s1|)
        diff_mat_max = np.max(diff_mat, axis=-1)  # (M2, M2)
        return np.intersect1d(
            a2actions, np.array(list(set(np.where(diff_mat_max < 0)[0])))
        )
    else:
        raise NotImplemented


def tgs_var_utility(init_points, constraints, bounds):
    """
    Runs the variable utility nonlinear feasibility problem on an array of initial points to find potential MNEs.
    :param init_points: array of shape (c, M1 + M2 + 2 + 2*(M1 * M2)).
    :param constraints: Tuple of constraint dicts.
    :param bounds: Tuple of bounds pairs.
    :return: (True, parameters) if a feasible set of parameters is found. Otherwise, returns (False, None).
    """
    fun = lambda x: 0  # feasibility problem

    for x0 in init_points:
        res = minimize(
            fun=fun,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False},
        )
        if res.success:
            return True, res.x

    return False, None


def tgs(constraints, bounds):
    x0 = np.zeros(len(bounds))
    fun = lambda x: 0  # feasibility problem
    res = minimize(
        fun=fun, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    if res.success:
        return True, res.x
    else:
        return False, None


def build_init_points(
    s1, s2, U1upper, U1lower, U2upper, U2lower, num_rand_dists_per_agent, rng
):
    """
    Constructs an array of initial points for the nonlinear feasibility problem.
    :param s1: 1-D array with length from 1 to M. Indices of actions in agent 1's support, ints from 0 - M-1.
    :param s2: 1-D array with length from 1 to M. Indices of actions in agent 2's support, ints from 0 - M-1.
    :param U1upper: Array of shape (M1, M2). Agent 1's upper bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U1lower: Array of shape (M1, M2). Agent 1's lower bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U2upper: Array of shape (M1, M2). Agent 2's upper bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U2lower: Array of shape (M1, M2). Agent 1's lower bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param num_rand_dists_per_agent: int.
    :param rng: NumPy rng object.
    :return: array of shape (c, M1 + M2 + 2 + 2*(M1 * M2)). c initial points to pass to nonlinear feasibility
    program to find potential MNEs. c depends on specific implementation + num_rand_dists_per_agent
    """
    M1, M2 = U1upper.shape
    # Indices
    u1start = M1 + M2 + 2
    u1end = u1start + M1 * M2 - 1
    u2start = u1start + M1 * M2
    u2end = u2start + M2 * M1 - 1
    p1utility = M1
    p2utility = M1 + M2 + 1
    p1start = 0
    p1end = M1 - 1
    p2start = M1 + 1
    p2end = M1 + M2

    U1mean = (U1lower + U1upper) / 2
    U2mean = (U2lower + U2upper) / 2
    p1supplength = len(s1)
    p2supplength = len(s2)

    init_points = []
    for i in range(3):
        if i == 0:
            U1source = U1mean
        elif i == 1:
            U1source = U1lower
        elif i == 2:
            U1source = U1upper
        curr_u1 = U1source.ravel()
        for j in range(3):
            if j == 0:
                U2source = U2mean
            elif j == 1:
                U2source = U2lower
            elif j == 2:
                U2source = U2upper
            U2source = U2source.T
            curr_u2 = U2source.ravel()
            for k in range(num_rand_dists_per_agent + 1):
                if k == 0:
                    curr_p1_prob = np.array(
                        [1 / p1supplength for _ in range(p1supplength)]
                    )
                else:
                    curr_p1_prob = unif_in_simplex(p1supplength, rng)
                for l in range(num_rand_dists_per_agent + 1):
                    if l == 0:
                        curr_p2_prob = np.array(
                            [1 / p2supplength for _ in range(p2supplength)]
                        )
                    else:
                        curr_p2_prob = unif_in_simplex(p2supplength, rng)
                    x = np.zeros(M1 + M2 + 2 + 2 * (M1 * M2))
                    x[np.arange(p1start, p1end + 1)[s1]] = curr_p1_prob
                    x[np.arange(p2start, p2end + 1)[s2]] = curr_p2_prob
                    curr_p1_prob_all = x[p1start : p1end + 1]
                    curr_p2_prob_all = x[p2start : p2end + 1]
                    x[p1utility] = curr_p1_prob_all @ U1source @ curr_p2_prob_all
                    x[p2utility] = curr_p2_prob_all @ U2source @ curr_p1_prob_all
                    x[u1start : u1end + 1] = curr_u1
                    x[u2start : u2end + 1] = curr_u2
                    init_points.append(x)

                    # if only 1 action in support, no point as all samples are 1 at that action
                    if p2supplength == 1:
                        break

                # if only 1 action in support, no point as all samples are 1 at that action
                if p1supplength == 1:
                    break

    return np.array(init_points)


def get_start_end(row, agent, M1, M2):
    """

    :param row: int. from 0 - (M1-1) if agent == 1, 0 - (M2-1) if agent == 2.
    :param agent: int. either 1 or 2.
    :param M1: int. Total number of actions for player 1.
    :param M2: int. Total number of actions for player 2.
    :return: tuple of ints.
    """
    u1start = M1 + M2 + 2
    u2start = u1start + M1 * M2
    if agent == 1:
        return u1start + row * M2, u1start + (row + 1) * M2 - 1
    elif agent == 2:
        return u2start + row * M1, u2start + (row + 1) * M1 - 1
    else:
        raise Exception("agent can only be 1 or 2")


def build_tgs_constraints_var_utility(s1, s2, U1upper, U1lower, U2upper, U2lower):
    """
    Builds constraint dictionaries for feasibility program TGS with variable utilities. Assumes variable x is an array
    of length 2 * (M + 1), where x[0] to x[M1-1] are the probabilities of each action for agent 1, x[M1] is the utility
    that agent 1 achieves, x[M1+1] to x[M1+M2] are the probabilities of each action for agent 2, and x[M1+M2+1] is the
    utility that agent 2 achieves.
    :param s1: 1-D array with length from 1 to M. Indices of actions in agent 1's support, ints from 0 - M-1.
    :param s2: 1-D array with length from 1 to M. Indices of actions in agent 2's support, ints from 0 - M-1.
    :param U1upper: Array of shape (M1, M2). Agent 1's upper bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U1lower: Array of shape (M1, M2). Agent 1's lower bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U2upper: Array of shape (M1, M2). Agent 2's upper bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :param U2lower: Array of shape (M1, M2). Agent 1's lower bound utility matrix. Agent 1 row player, agent 2 column
    player.
    :return: Tuple of constraint dicts.
    """
    M1, M2 = U1upper.shape
    s2_shifted = s2 + (M1 + 1)
    s1_c = np.setdiff1d(np.arange(0, M1), s1)
    s2_c = np.setdiff1d(np.arange(0, M2), s2)
    s2_c_shifted = s2_c + (M1 + 1)
    s1_ones = np.ones(len(s1))
    s2_ones = np.ones(len(s2))

    constraints = []
    # Eq. (4.26)
    for a1 in s1:

        def create_func(start, end):
            s = start
            e = end
            return lambda x: x[s : e + 1] @ x[M1 + 1 : M1 + M2 + 1] - x[M1]

        start, end = get_start_end(row=a1, agent=1, M1=M1, M2=M2)
        # constraints.append({'type': 'eq', 'fun': lambda x: x[start:end + 1] @ x[M1 + 1:M1 + M2 + 1] - x[M1]})
        constraints.append({"type": "eq", "fun": create_func(start, end)})
    for a2 in s2:

        def create_func(start, end):
            s = start
            e = end
            return lambda x: x[s : e + 1] @ x[:M1] - x[M1 + M2 + 1]

        start, end = get_start_end(row=a2, agent=2, M1=M1, M2=M2)
        # constraints.append({'type': 'eq', 'fun': lambda x: x[start:end + 1] @ x[:M1] - x[M1 + M2 + 1]})
        constraints.append({"type": "eq", "fun": create_func(start, end)})

    # Eq. (4.27)
    for a1 in s1_c:

        def create_func(start, end):
            s = start
            e = end
            return lambda x: x[M1] - x[s : e + 1] @ x[M1 + 1 : M1 + M2 + 1]

        start, end = get_start_end(row=a1, agent=1, M1=M1, M2=M2)
        # constraints.append({'type': 'ineq', 'fun': lambda x: x[M1] - x[start:end + 1] @ x[M1 + 1:M1 + M2 + 1]})
        constraints.append({"type": "ineq", "fun": create_func(start, end)})
    for a2 in s2_c:

        def create_func(start, end):
            s = start
            e = end
            return lambda x: x[M1 + M2 + 1] - x[s : e + 1] @ x[:M1]

        start, end = get_start_end(row=a2, agent=2, M1=M1, M2=M2)
        # constraints.append({'type': 'ineq', 'fun': lambda x: x[M1 + M2 + 1] - x[start:end + 1] @ x[:M1]})
        constraints.append({"type": "ineq", "fun": create_func(start, end)})

    # Eq. (4.29)
    constraints.append({"type": "eq", "fun": lambda x: x[s1_c]})
    constraints.append({"type": "eq", "fun": lambda x: x[s2_c_shifted]})
    # Eq. (4.30)
    constraints.append({"type": "eq", "fun": lambda x: x[s1] @ s1_ones - 1.0})
    constraints.append({"type": "eq", "fun": lambda x: x[s2_shifted] @ s2_ones - 1.0})

    # bounds
    bounds = (
        [(0, None) for _ in range(M1)]
        + [(None, None)]
        + [(0, None) for _ in range(M2)]
        + [(None, None)]
    )
    bounds += list(zip(list(U1lower.ravel()), list(U1upper.ravel())))
    bounds += list(zip(list(U2lower.T.ravel()), list(U2upper.T.ravel())))

    return tuple(constraints), tuple(bounds)


def build_tgs_constraints(s1, s2, U1, U2):
    """
    Builds constraint dictionaries for feasibility program TGS with known utilities. Assumes variable x is an array
    of length 2 * (M + 1), where x[0] to x[M1-1] are the probabilities of each action for agent 1, x[M1] is the utility
    that agent 1 achieves, x[M1+1] to x[M1+M2] are the probabilities of each action for agent 2, and x[M1+M2+1] is the
    utility that agent 2 achieves.
    :param s1: 1-D array with length from 1 to M. Indices of actions in agent 1's support, ints from 0 - M-1.
    :param s2: 1-D array with length from 1 to M. Indices of actions in agent 2's support, ints from 0 - M-1.
    :param U1: Array of shape (M1, M2). Agent 1's utility matrix. Agent 1 row player, agent 2 column player.
    :param U2: Array of shape (M1, M2). Agent 2's utility matrix. Agent 1 row player, agent 2 column player.
    :return: Tuple of constraint dicts.
    """
    M1, M2 = U1.shape
    s2_shifted = s2 + (M1 + 1)
    s1_c = np.setdiff1d(np.arange(0, M1), s1)
    s2_c = np.setdiff1d(np.arange(0, M2), s2)
    s2_c_shifted = s2_c + (M1 + 1)
    s1_ones = np.ones(len(s1))
    s2_ones = np.ones(len(s2))
    # Eq. (4.26)
    c1 = {"type": "eq", "fun": lambda x: U1[s1] @ x[M1 + 1 : M1 + M2 + 1] - x[M1]}
    c2 = {"type": "eq", "fun": lambda x: U2.T[s2] @ x[:M1] - x[M1 + M2 + 1]}
    # Eq. (4.27)
    c3 = {"type": "ineq", "fun": lambda x: x[M1] - U1[s1_c] @ x[M1 + 1 : M1 + M2 + 1]}
    c4 = {"type": "ineq", "fun": lambda x: x[M1 + M2 + 1] - U2.T[s2_c] @ x[:M1]}
    # Eq. (4.29)
    c5 = {"type": "eq", "fun": lambda x: x[s1_c]}
    c6 = {"type": "eq", "fun": lambda x: x[s2_c_shifted]}
    # Eq. (4.30)
    c7 = {"type": "eq", "fun": lambda x: x[s1] @ s1_ones - 1.0}
    c8 = {"type": "eq", "fun": lambda x: x[s2_shifted] @ s2_ones - 1.0}

    # bounds
    bounds = tuple(
        [(0, None) for _ in range(M1)]
        + [(None, None)]
        + [(0, None) for _ in range(M2)]
        + [(None, None)]
    )

    return (c1, c2, c3, c4, c5, c6, c7, c8), bounds
