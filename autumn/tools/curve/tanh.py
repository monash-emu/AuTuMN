from numpy import tanh


def tanh_based_scaleup(shape, inflection_time, lower_asymptote, upper_asymptote=1.0):
    """
    return the function t: (1 - sigma) / 2 * tanh(b * (a - c)) + (1 + sigma) / 2
    :param shape: shape parameter
    :param inflection_time: inflection point
    :param start_asymptote: lowest asymptotic value
    :param end_asymptote: highest asymptotic value
    :return: a function
    """
    rng = upper_asymptote - lower_asymptote

    def tanh_scaleup(t, cv=None):
        return (tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + lower_asymptote

    return tanh_scaleup
