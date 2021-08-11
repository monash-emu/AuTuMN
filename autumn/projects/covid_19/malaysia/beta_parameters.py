import math
import os
from typing import Any, Dict, List

import numpy as np
import yaml
from pyDOE import lhs
from scipy import special, stats
from scipy.optimize import minimize

def find_distribution_params_from_mean_and_ci(distribution, mean, ci, ci_width=0.95):
    """
    Work out the parameters of a given distribution based on a desired mean and CI
    :param distribution: string
        either 'gamma' or 'beta' for now
    :param mean: float
        the desired mean value
    :param ci: list or tuple of length 2
        the lower and upper bounds of the CI
    :param ci_width:
        the width of the desired CI
    :return:
        a dictionary with the parameters
    """
    assert len(ci) == 2 and ci[1] > ci[0] and 0.0 < ci_width < 1.0
    percentile_low = (1.0 - ci_width) / 2.0
    percentile_up = 1.0 - percentile_low

    if distribution == "beta":
        assert 0.0 < ci[0] < 1.0 and 0.0 < ci[1] < 1.0 and 0.0 < mean < 1.0

        def distance_to_minimise(a):
            b = a * (1.0 - mean) / mean
            vals = stats.beta.ppf([percentile_low, percentile_up], a, b)
            dist = sum([(ci[i] - vals[i]) ** 2 for i in range(2)])
            return dist

        sol = minimize(distance_to_minimise, [1.0], bounds=[(0.0, None)], tol=1.0e-32)
        best_a = sol.x
        best_b = best_a * (1.0 - mean) / mean
        params = {"a": best_a, "b": best_b}

    elif distribution == "gamma":
        assert ci[0] > 0 and ci[1] > 0 and mean > 0.0

        def distance_to_minimise(scale):
            shape = mean / scale
            vals = stats.gamma.ppf([percentile_low, percentile_up], shape, 0, scale)
            dist = sum([(ci[i] - vals[i]) ** 2 for i in range(2)])
            return dist

        sol = minimize(distance_to_minimise, [1.0], bounds=[(1.0e-11, None)])
        best_scale = sol.x
        best_shape = mean / best_scale

        params = {"shape": best_shape, "scale": best_scale}
    else:
        raise ValueError(distribution + " distribution is not supported for the moment")

    return params


vals = find_distribution_params_from_mean_and_ci("beta",0.7,[0.5,0.9],ci_width=0.95)
print(vals)
