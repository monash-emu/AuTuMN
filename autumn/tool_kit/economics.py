"""
Utility functions for economic calculations.
"""
import numpy


def get_cost_from_coverage(
    coverage, saturation, unit_cost, popsize, inflection_cost=0.0, alpha=1.0
):
    """
    Estimate the raw cost associated with a given coverage for a specific intervention.

    Args:
        coverage: The intervention's coverage as a proportion
        inflection_cost: Cost at which inflection occurs on the curve (also the point of maximal efficiency)
        saturation: Maximal possible coverage, i.e. upper asymptote of the logistic curve
        unit_cost: Unit cost of the intervention
        popsize: Size of the population targeted by the intervention
        alpha: Steepness parameter determining the curve's shape
    Returns:
        The raw cost from the logistic function
    """

    # if unit cost or pop_size or coverage is null, return 0
    if popsize == 0.0 or unit_cost == 0.0 or coverage == 0.0:
        return 0.0
    # assert 0. <= coverage <= saturation, 'Coverage must satisfy 0 <= coverage <= saturation'
    if coverage == saturation:
        coverage = saturation - 1e-6

    # logistic curve function code
    a = saturation / (1.0 - 2.0 ** alpha)
    b = 2.0 ** (alpha + 1.0) / (alpha * (saturation - a) * unit_cost * popsize)
    return inflection_cost - 1.0 / b * numpy.log(
        ((saturation - a) / (coverage - a)) ** (1.0 / alpha) - 1.0
    )


def get_coverage_from_cost(
    spending, saturation, unit_cost, popsize, inflection_cost=0.0, alpha=1.0
):
    """
    Estimate the coverage associated with a spending in a program.

    Args:
        spending: The amount of money allocated to a program (absolute value, not a proportion of all funding)
        inflection_cost: Cost at which inflection occurs on the curve (also the point of maximal efficiency)
        saturation: Maximal possible coverage, i.e. upper asymptote of the logistic curve
        unit_cost: Unit cost of the intervention
        popsize: Size of the population targeted by the intervention
        alpha: Steepness parameter determining the curve's shape
    Returns:
        The proportional coverage of the intervention given the spending
   """

    # if cost is smaller thar c_inflection_cost, then the starting cost necessary to get coverage has not been reached
    if popsize == 0.0 or unit_cost == 0.0 or spending == 0.0 or spending <= inflection_cost:
        return 0.0

    a = saturation / (1.0 - 2.0 ** alpha)
    b = 2.0 ** (alpha + 1.0) / (alpha * (saturation - a) * unit_cost * popsize)
    return a + (saturation - a) / ((1.0 + numpy.exp((-b) * (spending - inflection_cost))) ** alpha)
