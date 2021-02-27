from math import tanh


def tanh_based_scaleup(max_gradient, inflection_time, lower_asymptote, upper_asymptote=1.):
    """
    return the function t: (1 - sigma) / 2 * tanh(b * (a - c)) + (1 + sigma) / 2
    :param max_gradient: shape parameter
    :param inflection_time: inflection point
    :param lower_asymptote: lowest asymptotic value
    :param upper_asymptote: highest asymptotic value
    :return: a function
    """
    assert upper_asymptote > lower_asymptote, "Lower asymptote is greater than upper asymptote"
    range = upper_asymptote - lower_asymptote

    def tanh_scaleup(t):
        return (tanh(max_gradient * (t - inflection_time)) / 2. + 0.5) * \
               range + \
               lower_asymptote

    return tanh_scaleup
