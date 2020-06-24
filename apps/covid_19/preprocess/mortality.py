from math import exp
from autumn.curve import numerical_integration


def age_specific_ifrs_from_double_exp_model(k, m, representative_age_oldest_group):
    """
    Work out age-specific IFR values assuming they are derived by the double exponential model:
    x -> exp(-a.exp(-b.x))
    :params a, b: parameters of the double exponential model
    :param representative_age_oldest_group: the age to be used to represent the oldest age category
    :return: a list of values representing the IFR for each age group using 5 year brackets
    """
    ifr_list = []
    for age_index in range(16):
        if age_index < 15:
            ifr_list.append(
                numerical_integration(lambda x: exp(-k * exp(-m * x)),
                                      age_index * 5., (age_index + 1) * 5.) / 5.
            )
        else:
            ifr_list.append(
                exp(-k * exp(-m * representative_age_oldest_group))
            )
    return ifr_list
