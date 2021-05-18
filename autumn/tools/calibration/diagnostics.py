import statsmodels.api as sm
from math import floor


def calculate_effective_sample_size(x):
    """
    Computes the effective sample size of a sample.
    x is a list containing the series of values.
    ESS calculation is similar to what is implemented in Stan (description here:
    https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html
    )
    """
    n = len(x)
    # compute autocorrelations
    autocor = sm.tsa.stattools.acf(x, nlags=n)[1:]

    # identify m (see Stan's notation)
    paired_sum = [autocor[2 * i] + autocor[2 * i + 1] for i in range(floor(len(autocor)/2))]
    first_neg_index = first_neg(paired_sum)
    if first_neg_index is None:
        m = len(paired_sum) - 1
    elif first_neg_index == 0:
        m = 0
    else:
        m = first_neg_index - 1

    tau = 1 + 2 * sum(autocor[1: (2 * m + 1)])
    ess = n / tau

    return ess


def first_neg(x):
    """
    Finds index of first negative number in a list
    """
    res = [i for i, x in enumerate(x) if x < 0]
    return None if res == [] else res[0]
