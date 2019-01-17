
# external imports
import scipy
import numpy


def get_cost_from_coverage(coverage, inflection_cost, saturation, unit_cost, popsize, alpha=1.):
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
    if popsize == 0. or unit_cost == 0. or coverage == 0.: return 0.
    #assert 0. <= coverage <= saturation, 'Coverage must satisfy 0 <= coverage <= saturation'
    if coverage == saturation: coverage = saturation - 1e-6

    # logistic curve function code
    a = saturation / (1. - 2. ** alpha)
    b = 2. ** (alpha + 1.) / (alpha * (saturation - a) * unit_cost * popsize)
    return inflection_cost - 1. / b * scipy.log(((saturation - a) / (coverage - a)) ** (1. / alpha) - 1.)


def get_coverage_from_cost(spending, inflection_cost, saturation, unit_cost, popsize, alpha=1.):
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
    if popsize == 0. or unit_cost == 0. or spending == 0. or spending <= inflection_cost: return 0.

    a = saturation / (1. - 2. ** alpha)
    b = 2. ** (alpha + 1.) / (alpha * (saturation - a) * unit_cost * popsize)
    return a + (saturation - a) / ((1. + numpy.exp((-b) * (spending - inflection_cost))) ** alpha)


def inflate_cost(raw_cost, current_cpi, cpi_time_variant):
    """
    Calculate the inflated cost associated with raw_cost, considering the current CPI and the CPI corresponding
    to the date considered (cpi_time_variant).

    Returns:
        The inflated cost
    """

    return raw_cost * current_cpi / cpi_time_variant


def discount_cost(raw_cost, discount_rate, t_into_future):
    """
    Calculate the discounted cost associated with the raw cost at time (t + t_into_future)

    Args:
        raw_cost: Cost without accounting for discounting
        discount_rate: Discounting rate per year
        t_into_future: Number of years into future at which we want to calculate the discounted cost
    Returns:
        The discounted cost
    """

    assert t_into_future >= 0., 't_into_future must be non-negative'
    return raw_cost / ((1. + discount_rate) ** t_into_future)


def get_adjusted_cost(raw_cost, approach, current_cpi=None, cpi_time_variant=None, discount_rate=None,
                      time_into_future=None):
    """
    Calculate the adjusted cost corresponding to a given type.

    Args:
        raw_cost: The raw cost
        approach: A string which must be one of 'inflated', 'discounted' or 'discounted_inflated'
        current_cpi: Consumer price index at the current time
        cpi_time_variant:
        discount_rate: Discounting rate per year
        time_into_future: Time into the future from the point at which the CPI was calculated
    Returns:
        The adjusted cost
    """

    if approach == 'inflated':
        return inflate_cost(raw_cost, current_cpi, cpi_time_variant)
    elif approach == 'discounted':
        return discount_cost(raw_cost, discount_rate, time_into_future)
    elif approach == 'discounted_inflated':
        return discount_cost(inflate_cost(raw_cost, current_cpi, cpi_time_variant), discount_rate, time_into_future)
