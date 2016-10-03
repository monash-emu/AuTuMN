
import scipy


def get_cost_from_coverage(coverage, c_inflection_cost, saturation, unit_cost, pop_size, alpha=1.):

    """
    Estimate the global uninflated cost associated with a given coverage
    Args:
        coverage: the coverage (as a proportion, then lives in 0-1)
        c_inflection_cost: cost at which inflection occurs on the curve. It's also the configuration leading to the
                            best efficiency.
        saturation: maximal acceptable coverage, ie upper asymptote
        unit_cost: unit cost of the intervention
        pop_size: size of the population targeted by the intervention
        alpha: steepness parameter

    Returns:
        uninflated cost

    """

    # If unit cost or pop_size is null, return 0
    if pop_size * unit_cost == 0.:
        return 0.
    assert 0. <= coverage < saturation, 'coverage must verify 0 <= coverage < saturation'

    if coverage == 0.:
        return 0.
    # Here's the logistic curve function code
    a = saturation / (1. - 2. ** alpha)
    b = ((2. ** (alpha + 1.)) / (alpha * (saturation - a) * unit_cost * pop_size))
    cost_uninflated = c_inflection_cost - 1. / b * scipy.log(
        (((saturation - a) / (coverage - a)) ** (1. / alpha)) - 1.)

    return cost_uninflated


def inflate_cost(cost_uninflated, current_cpi, cpi_time_variant):

    """
    Calculate the inflated cost associated with cost_uninflated and considering the current cpi and the cpi correponding
    to the date considered (cpi_time_variant)

    Returns:
        the inflated cost
    """

    return cost_uninflated * current_cpi / cpi_time_variant


def discount_cost(cost_uninflated, discount_rate, t_into_future):

    """
    Calculate the discounted cost associated with cost_uninflated at time (t + t_into_future)
    Args:
        cost_uninflated: cost without accounting for discounting
        discount_rate: discount rate (/year)
        t_into_future: number of years into future at which we want to calculate the discounted cost

    Returns:
        the discounted cost
    """

    assert t_into_future >= 0, 't_into_future must be >= 0'
    return cost_uninflated / ((1 + discount_rate) ** t_into_future)


def get_adjusted_cost(raw_cost, type, current_cpi=None, cpi_time_variant=None,discount_rate=None, t_into_future=None):
    """
    calculate the adjusted cost corresponding to a given type
    Args:
        raw_cost: the raw cost
        type: one of ['inflated', 'discounted', 'discounted_inflated']

    Returns: the adjusted cost
    """
    if type == 'inflated':
        return inflate_cost(raw_cost, current_cpi, cpi_time_variant)
    elif type == 'discounted':
        return discount_cost(raw_cost, discount_rate, t_into_future)
    elif type == 'discounted_inflated':
        inflated_cost = inflate_cost(raw_cost, current_cpi, cpi_time_variant)
        return discount_cost(inflated_cost, discount_rate, t_into_future)
