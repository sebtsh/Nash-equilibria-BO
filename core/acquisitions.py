import numpy as np

from core.ne import best_response_payoff_pure


def create_ucb_funcs(models,
                     beta):
    """
    Converts GP models into UCB functions.
    :param models: List of N GPflow GPs.
    :param beta: float.
    :return: List of Callables that take in an array of shape (n, N) and return an array of shape (n, 1).
    """
    N = len(models)

    def create_ucb_func(model):
        def inn(X):
            mean, var = model.predict_f(X)
            return mean + beta * np.sqrt(var)
        return inn

    return [create_ucb_func(models[i]) for i in range(N)]


def ucb_ne(beta,
           domain,
           actions):
    def acq(models):
        """
        Returns a point to query next.
        :param models: List of N GPflow GPs.
        :return: array of shape (1, N).
        """
        ucb_funcs = create_ucb_funcs(models=models,
                                     beta=beta)
        ucb_brp = best_response_payoff_pure(u=ucb_funcs,
                                            S=domain,
                                            actions=actions)  # array of shape (M ** N, N)
        next_idx = np.argmin(np.max(ucb_brp, axis=-1))
        return domain[next_idx:next_idx + 1]

    return acq
