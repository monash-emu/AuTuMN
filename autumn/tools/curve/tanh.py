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


def step_based_scaleup_for_notifications(magnitude_low, magnitude_high,  time_in_effect, contact_rate, t):
    if t < time_in_effect:
        microdistancing_effect = magnitude_low
    else:
        microdistancing_effect = magnitude_high

    contact_rate = contact_rate * (microdistancing_effect ** 2)  # adjust transmission rate
    # to have microdistancing effect on notifications

    return contact_rate





