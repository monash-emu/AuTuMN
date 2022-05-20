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


def step_based_scaleup_for_notifications(magnitude, time_in_effect, start, end):
    low_microdistancing = [0] * len(list(range(start, time_in_effect+1)))  # from start to time in effect low microdistncing
    high_microdistancing = [magnitude] * len(list(range(time_in_effect, end+1)))  # from time in effect to end high microdistncing
    microdistancing_effect = low_microdistancing + high_microdistancing
    return microdistancing_effect





