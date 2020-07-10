from math import exp, tanh
from autumn.curve import numerical_integration


def age_specific_ifrs_from_double_exp_model(k, m, representative_age_oldest_group):
    """
    Work out age-specific IFR values assuming they are derived by the double exponential model:
    x -> exp(-k.exp(-m.x))
    :params k, m: parameters of the double exponential model
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


def age_specific_ifrs_from_exp_tanh_model(k, b, c, representative_age_oldest_group):
    """
    Work out age-specific IFR values assuming they are derived by the model:
    x -> exp(-k.(0.5tanh(b(x-c)) + 0.5))
    :params k, b, c: parameters of the double exponential model
    :param representative_age_oldest_group: the age to be used to represent the oldest age category
    :return: a list of values representing the IFR for each age group using 5 year brackets
    """
    ifr_list = []
    for age_index in range(16):
        if age_index < 15:
            ifr_list.append(
                numerical_integration(
                    lambda x: exp(-k*(0.5*tanh(b*(x-c)) + 0.5)),
                    age_index * 5., (age_index + 1) * 5.)
                / 5.
            )
        else:
            ifr_list.append(
                exp(-k*(0.5*tanh(b*(representative_age_oldest_group-c)) + 0.5))
            )
    return ifr_list
