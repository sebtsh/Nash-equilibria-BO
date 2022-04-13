import numpy as np
from scipy.optimize import minimize
from core.mne import (
    build_tgs_constraints,
    build_tgs_constraints_var_utility,
    build_init_points,
    tgs,
    tgs_var_utility,
    SEM,
    SEM_var_utility,
    get_strategies_and_support,
)

rng = np.random.default_rng(0)
M1 = 3
M2 = 2
N = 2  # num_players

U1upper = np.array([[0.0, 6.0], [2.0, 5.0], [3.0, 3.0]]) + 0.2
U1lower = np.array([[0.0, 6.0], [2.0, 5.0], [3.0, 3.0]]) - 0.2
U2upper = np.array([[2.1, 0.0], [0.0, 2.0], [6.1, 3.0]]) + 0.2
U2lower = np.array([[2.1, 0.0], [0.0, 2.0], [5.1, 3.0]]) - 0.2

# print("U1upper")
# print(U1upper)
# print("U1lower")
# print(U1lower)
# print("U2upper")
# print(U2upper)
# print("U2lower")
# print(U2lower)

# from core.utils import sort_size_balance
# import itertools
# from core.mne import conditionally_dominated_var_utility
# pairs = [(x, y) for x in range(1, M1 + 1) for y in range(1, M2 + 1)]
# sorted_pairs = sort_size_balance(pairs)
# mnes = []
# for pair in sorted_pairs:
#     s1_size, s2_size = pair
#     all_s1 = itertools.combinations(np.arange(M1), s1_size)
#     for s1 in all_s1:
#         all_s2 = itertools.combinations(np.arange(M2), s2_size)
#         for s2 in all_s2:
#             print(f"s1:{s1}, s2:{s2}")
#             print("Agent 1:")
#             print(conditionally_dominated_var_utility(p1actions=s1,
#                                                       p2actions=s2,
#                                                       active_agent=1,
#                                                       U1upper=U1upper,
#                                                       U1lower=U1lower,
#                                                       U2upper=U2upper,
#                                                       U2lower=U2lower))
#             print("Agent 2:")
#             print(conditionally_dominated_var_utility(p1actions=s1,
#                                                       p2actions=s2,
#                                                       active_agent=2,
#                                                       U1upper=U1upper,
#                                                       U1lower=U1lower,
#                                                       U2upper=U2upper,
#                                                       U2lower=U2lower))
#             print("=======")

all_res = SEM_var_utility(
    U1upper,
    U1lower,
    U2upper,
    U2lower,
    num_rand_dists_per_agent=5,
    rng=rng,
    mode="all",
)
u1start = M1 + M2 + 2
u2start = u1start + M1 * M2
for res in all_res:
    (s1, s2), (a1supp, a2supp) = get_strategies_and_support(res, M1, M2)
    print((s1, s2), (a1supp, a2supp))
    print(f"Agent 1 strategy: {res[:M1]}, with value {res[M1]}")
    print(f"Agent 2 strategy: {res[M1 + 1:M1 + M2 + 1]}, with value {res[M1 + M2 + 1]}")
    print(f"U1: {np.reshape(res[u1start:u1start + M1 * M2], (M1, M2))}")
    print(f"U2: {np.reshape(res[u2start:u2start + M2 * M1], (M2, M1)).T}")
    print("==============")

####################################################

# rng = np.random.default_rng(0)
# M1 = 3
# M2 = 2
# N = 2  # num_players
#
# U1 = np.array([[0.0, 6.0], [2.0, 5.0], [3.0, 3.0]])
# U2 = np.array([[1.0, 0.0], [0.0, 2.0], [4.0, 3.0]])
#
# # M1 = 3
# # M2 = 3
# # N = 2  # num_players
# #
# # U1 = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1], [1.0, -1.0, 0]])
# # U2 = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1], [-1.0, 1.0, 0]])
#
# all_res = SEM(M1, M2, U1, U2)
#
# for res in all_res:
#     print(f"Agent 1 strategy: {res[:M1]}, with value {res[M1]}")
#     print(f"Agent 2 strategy: {res[M1 + 1:M1 + M2 + 1]}, with value {res[M1 + M2 + 1]}")
#     print("==============")


########################################################

# U1lower = np.array([[0.0, 6.0], [2.0, 5.0], [3.0, 3.0]]) - 0.01
# U1upper = np.array([[0.0, 6.0], [2.0, 5.0], [3.0, 3.0]]) + 0.01
# U2lower = np.array([[1.0, 0.0], [0.0, 2.0], [4.0, 3.0]]) - 0.01
# U2upper = np.array([[1.0, 0.0], [0.0, 2.0], [4.0, 3.0]]) + 0.01
#
# # U1lower = np.zeros((3, 2)) - rng.normal(0.5, 0.5, (3, 2))
# # U1upper = np.ones((3, 2)) + rng.normal(0.5, 0.5, (3, 2))
# # U2lower = np.zeros((3, 2)) - rng.normal(0.5, 0.5, (3, 2))
# # U2upper = np.ones((3, 2)) + rng.normal(0.5, 0.5, (3, 2))
#
# s1 = np.array([1, 2])
# s2 = np.array([0, 1])
# cons, bounds = build_tgs_constraints_var_utility(
#     M1=M1,
#     M2=M2,
#     s1=s1,
#     s2=s2,
#     U1upper=U1upper,
#     U1lower=U1lower,
#     U2upper=U2upper,
#     U2lower=U2lower,
# )
#
# init_points = build_init_points(
#     M1=M1,
#     M2=M2,
#     s1=s1,
#     s2=s2,
#     U1upper=U1upper,
#     U1lower=U1lower,
#     U2upper=U2upper,
#     U2lower=U2lower,
#     num_rand_dists_per_agent=5,
#     rng=rng,
# )
#
# is_success, x = tgs_var_utility(
#     init_points=init_points, constraints=cons, bounds=bounds
# )
#
# print(f"Success: {is_success}")
#
# # x0 = (
# #     [0.0, 0.5, 0.5, np.mean(U1lower), 0.5, 0.5, np.mean(U2lower)]
# #     + list(((U1lower + U1upper) / 2).ravel())
# #     + list(((U2lower.T + U2upper.T) / 2).ravel())
# # )
# # fun = lambda x: 0  # feasibility problem
# #
# # res = minimize(
# #     fun=fun,
# #     x0=x0,
# #     method="SLSQP",
# #     bounds=bounds,
# #     constraints=cons,
# #     options={"disp": False},
# # )
# #
# # print(res)
# # x = res.x
# u1start = M1 + M2 + 2
# u2start = u1start + M1 * M2
# print(f"Agent 1 strategy: {x[:M1]}, with value {x[M1]}")
# print(f"Agent 2 strategy: {x[M1 + 1:M1 + M2 + 1]}, with value {x[M1 + M2 + 1]}")
# print(f"U1: {np.reshape(x[u1start:u1start + M1 * M2], (M1, M2))}")
# print(f"U2: {np.reshape(x[u2start:u2start + M2 * M1], (M2, M1)).T}")

###################################################

# M1 = 3
# M2 = 2
# N = 2  # num_players

# new_U1 = np.reshape(x[u1start: u1start + M1 * M2], (M1, M2))
# new_U2 = np.reshape(x[u2start: u2start + M2 * M1], (M2, M1)).T

# U1 = np.array([[0., 6.],
#                [2., 5.],
#                [3., 3.]])
# U2 = np.array([[1., 0.],
#                [0., 2.],
#                [4., 3.]])
# s1 = np.array([0])
# s2 = np.array([0, 1])
# cons, bounds = build_tgs_constraints(M1=M1, M2=M2, s1=s1, s2=s2, U1=new_U1, U2=new_U2)
# success, res = tgs(cons, bounds, M1, M2)
# print(res)
