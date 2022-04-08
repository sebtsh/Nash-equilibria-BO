import numpy as np
from core.utils import unif_in_simplex


def build_init_points(
    M1, M2, s1, s2, U1upper, U1lower, U2upper, U2lower, num_rand_dists_per_agent, rng
):
    """

    :param M1: int. Total number of actions for player 1.
    :param M2: int. Total number of actions for player 2.
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
    :return: array of shape (c, M1 + M2 + 2 + 2*(M1 + M2)). c initial points to pass to nonlinear feasibility
    program to find potential MNEs. c depends on specific implementation + num_rand_dists_per_agent
    """
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
        curr_p1_util = np.mean(U1source)
        curr_u1 = U1source.ravel()
        for j in range(3):
            if j == 0:
                U2source = U2mean
            elif j == 1:
                U2source = U2lower
            elif j == 2:
                U2source = U2upper
            curr_p2_util = np.mean(U2source)
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
                    x[p1utility] = curr_p1_util
                    x[np.arange(p2start, p2end + 1)[s2]] = curr_p2_prob
                    x[p2utility] = curr_p2_util
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


def build_tgs_constraints(M1, M2, s1, s2, U1, U2):
    """
    Builds constraint dictionaries for feasibility program TGS with known utilities. Assumes variable x is an array
    of length 2 * (M + 1), where x[0] to x[M1-1] are the probabilities of each action for agent 1, x[M1] is the utility
    that agent 1 achieves, x[M1+1] to x[M1+M2] are the probabilities of each action for agent 2, and x[M1+M2+1] is the
    utility that agent 2 achieves.
    :param M1: int. Total number of actions for player 1.
    :param M2: int. Total number of actions for player 2.
    :param s1: 1-D array with length from 1 to M. Indices of actions in agent 1's support, ints from 0 - M-1.
    :param s2: 1-D array with length from 1 to M. Indices of actions in agent 2's support, ints from 0 - M-1.
    :param U1: Array of shape (M, M). Agent 1's utility matrix. Agent 1 row player, agent 2 column player.
    :param U2: Array of shape (M, M). Agent 2's utility matrix. Agent 1 row player, agent 2 column player.
    :return: Tuple of constraint dicts.
    """
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


def build_tgs_constraints_var_utility(
    M1, M2, s1, s2, U1upper, U1lower, U2upper, U2lower
):
    """
    Builds constraint dictionaries for feasibility program TGS with variable utilities. Assumes variable x is an array
    of length 2 * (M + 1), where x[0] to x[M1-1] are the probabilities of each action for agent 1, x[M1] is the utility
    that agent 1 achieves, x[M1+1] to x[M1+M2] are the probabilities of each action for agent 2, and x[M1+M2+1] is the
    utility that agent 2 achieves.
    :param M1: int. Total number of actions for player 1.
    :param M2: int. Total number of actions for player 2.
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
