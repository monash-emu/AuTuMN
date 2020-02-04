"""
Utilities for preprocessing inputs to the model.
"""


def convert_competing_proportion_to_rate(competing_flows):
    """
    convert a proportion to a rate dependent on the other flows coming out of a compartment
    """
    return lambda proportion: proportion * competing_flows / (1.0 - proportion)


def scale_relative_risks_for_equivalence(proportions, relative_risks):
    """
    :param proportions: dictionary
    :param relative_risks: dictionary
    :return: dictionary
    """
    new_reference_deno = 0.0
    for stratum in proportions.keys():
        new_reference_deno += proportions[stratum] * relative_risks[stratum]
    new_reference = 1.0 / new_reference_deno
    for stratum in relative_risks.keys():
        relative_risks[stratum] *= new_reference
    return relative_risks
