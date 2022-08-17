import itertools
import numpy as np
from scipy.optimize import minimize

from core.utils import sort_size_balance


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
