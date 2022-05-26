from math import log
from numpy import exp


def logit(u):
    return log(u / (1 - u))


def inverse_logit(v):
    return 1 / (1 + exp(-v))


def make_transform_func_with_lower_bound(lower_bound, func_type):
    assert func_type in ["direct", "inverse", "inverse_derivative"]
    if func_type == "direct":

        def func(x):
            return log(x - lower_bound)

    elif func_type == "inverse":

        def func(y):
            return exp(y) + lower_bound

    elif func_type == "inverse_derivative":

        def func(y):
            return exp(y)

    return func


def make_transform_func_with_upper_bound(upper_bound, func_type):
    assert func_type in ["direct", "inverse", "inverse_derivative"]
    if func_type == "direct":

        def func(x):
            return log(upper_bound - x)

    elif func_type == "inverse":

        def func(y):
            return upper_bound - exp(y)

    elif func_type == "inverse_derivative":

        def func(y):
            return exp(y)

    return func


def make_transform_func_with_two_bounds(lower_bound, upper_bound, func_type):
    assert func_type in ["direct", "inverse", "inverse_derivative"]
    if func_type == "direct":

        def func(x):
            return logit((x - lower_bound) / (upper_bound - lower_bound))

    elif func_type == "inverse":

        def func(y):
            # prevent overflow when the parameter gets close to its bounds (i.e. transformed param's norm gets too big)
            if y < -100.0:
                x = lower_bound
            elif y > 100.0:
                x = upper_bound
            else:
                x = lower_bound + (upper_bound - lower_bound) * inverse_logit(y)
            return x

    elif func_type == "inverse_derivative":

        def func(y):
            if y < -100.0:
                return 0.0
            elif y > 100.0:
                return 0.0
            else:
                inv_logit = inverse_logit(y)
                return (upper_bound - lower_bound) * inv_logit * (1 - inv_logit)

    return func
