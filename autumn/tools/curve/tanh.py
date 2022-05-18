from numpy import tanh


def tanh_based_scaleup(shape, inflection_time, start_asymptote, end_asymptote=1.0):
    """
    return the function t: (1 - sigma) / 2 * tanh(b * (a - c)) + (1 + sigma) / 2
    :param shape: shape parameter
    :param inflection_time: inflection point
    :param start_asymptote: lowest asymptotic value
    :param end_asymptote: highest asymptotic value
    :return: a function
    """
    rng = end_asymptote - start_asymptote

    def tanh_scaleup(t, cv=None):
        return (tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote

    return tanh_scaleup


def tanh_based_scaleup_for_notifications(notifications, shape, inflection_time, start_asymptote, end_asymptote=1.0):
    """
    return the function notifications: (1 - sigma) / 2 * tanh(b * (a - c)) + (1 + sigma) / 2
    :notifications: notified COVID-19 cases
    :param shape: shape parameter
    :param inflection_time: inflection point
    :param start_asymptote: lowest asymptotic value
    :param end_asymptote: highest asymptotic value
    :return: a function
    """
    rng = end_asymptote - start_asymptote

    def tanh_scaleup(cases, cv=None):
        return (tanh(shape * (cases - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote

    return tanh_scaleup(notifications)
