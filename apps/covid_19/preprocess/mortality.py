from math import exp


def age_specific_ifrs_from_double_exp_model(a, b, representative_age_oldest_group):
    """
    Work out age-specific IFR values assuming they are derived by the double exponential model:
    x -> exp(-a.exp(-b.x))
    :params a, b: parameters of the double exponential model
    :param representative_age_oldest_group: the age to be used to represent the oldest age category
    :return: a list of values representing the IFR for each age group using 5 year brackets
    """
    ifr_list = []
    for age_index in range(16):
        representative_age = age_index * 5 + 2.5 if age_index < 15 else representative_age_oldest_group
        ifr_list.append(
            exp(-a * exp(-b * representative_age))
        )

    return ifr_list
