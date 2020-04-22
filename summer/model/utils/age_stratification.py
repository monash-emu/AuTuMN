"""
Functions needed for dealing with age stratification
"""


def add_zero_to_age_breakpoints(breakpoints):
    """
    append a zero on to a list if there isn't one already present, for the purposes of age stratification

    :param breakpoints: list
        integers for the age breakpoints requested
    :return: list
        age breakpoints with the zero value included
    """
    return [0] + breakpoints if 0 not in breakpoints else breakpoints


def split_age_parameter(age_breakpoints, parameter):
    """
    creates a dictionary to request splitting of a parameter according to age breakpoints, but using values of 1 for
        each age stratum
    allows that later parameters that might be age-specific can be modified for some age strata

    :param age_breakpoints: list
        list of the age breakpoints to be requested, with breakpoints as string
    :param parameter: str
        name of parameter that will need to be split
    :return: dict
        dictionary with age groups as string as keys and ones for all the values
    """
    age_breakpoints = ["0"] + age_breakpoints if "0" not in age_breakpoints else age_breakpoints
    return {parameter: {str(age_group): 1.0 for age_group in age_breakpoints}}
