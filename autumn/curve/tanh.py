from math import tanh


def tanh_based_scaleup(b, c, sigma, upper_asymptote=1.0):
    """
    return the function t: (1 - sigma) / 2 * tanh(b * (a - c)) + (1 + sigma) / 2
    :param b: shape parameter
    :param c: inflection point
    :param sigma: lowest asymptotic value
    :param upper_asymptote: highest asymptotic value
    :return: a function
    """

    def tanh_scaleup(t):
        return (tanh(b * (t - c)) / 2.0 + 0.5) * (upper_asymptote - sigma) + sigma

    return tanh_scaleup
