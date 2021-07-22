import statsmodels.api as sm
from math import floor, sqrt
from numpy import mean


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


def calculate_r_hat(posterior_chains):
    """
    Calculate the R_hat statistic for a single parameter. The code below is intended to be compatible with chains of
    different lengths. This is why the calculations may look slightly different compared to what is found in classic
    textbooks.
    :param posterior_chains: a dictionary, The keys are the chains ids and the values contain each chain's posterior
    sample.
    :return: the R_hat statistic (float)
    """
    m = len(posterior_chains)

    # Compute within-chain means
    means_per_chain = {}
    for j_chain, x_j in posterior_chains.items():
        means_per_chain[j_chain] = mean(x_j)

    # Compute overall mean
    flat_listed_values = sum(list(posterior_chains.values()), [])
    overall_mean = mean(flat_listed_values)

    # Compute between-chain variation (B / n)
    b_over_n = 1 / (m - 1) * sum([(means_per_chain[j_chain] - overall_mean)**2 for j_chain in range(m)])

    # Compute within-chain variation for each chain
    variation, chain_length = {}, {}
    for j_chain, x_j in posterior_chains.items():
        n_j = len(x_j)
        variation[j_chain] = sum([(x_j[i] - means_per_chain[j_chain])**2 for i in range(n_j)])
        chain_length[j_chain] = n_j

    # Compute the average of within-chain variances (W)
    w = 1 / m * sum([1 / (chain_length[j_chain] - 1) * variation[j_chain] for j_chain in range(m)])

    # Calculate the marginal posterior variance
    var_hat = 1 / m * sum([1 / chain_length[j_chain] * variation[j_chain] for j_chain in range(m)]) + b_over_n

    # Calculate R_hat
    r_hat = sqrt(var_hat / w)

    return r_hat

