from math import tanh


def tanh_based_scaleup(shape, inflection_time, lower_asymptote, upper_asymptote=1.0):
    """
    return the function t: (1 - sigma) / 2 * tanh(b * (a - c)) + (1 + sigma) / 2
    :param shape: shape parameter
    :param inflection_time: inflection point
    :param lower_asymptote: lowest asymptotic value
    :param upper_asymptote: highest asymptotic value
    :return: a function
    """
    range = upper_asymptote - lower_asymptote
    assert range >= 0.0, "Lower asymptote is greater than upper asymptote"

    def tanh_scaleup(t):
        return (tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * range + lower_asymptote

    return tanh_scaleup
