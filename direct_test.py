from scipydirect import minimize as direct_minimize

bounds = [[-1.0, 1.0], [-1.0, 1.0]]


def obj(x):
    def inner(y):
        return x[0] ** 2 + x[0] * y + x[0] + y**2

    return -direct_minimize(inner, bounds=bounds[:1]).fun


res = direct_minimize(obj, bounds=bounds)
print(res)
