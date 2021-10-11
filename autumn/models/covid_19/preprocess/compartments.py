from typing import Dict

from autumn.models.covid_19.parameters import Sojourn


def calc_compartment_periods(sojourn: Sojourn) -> Dict[str, float]:
    """
    Calculate compartment periods from the provided splits into early and late components of the exposed and active
    periods.
    """

    final_periods = {**sojourn.compartment_periods}
    for calc_period_name, calc_period_def in sojourn.compartment_periods_calculated.items():
        total_props = 0.
        for comp_name, prop in calc_period_def.proportions.items():
            new_period_dict = {f"{comp_name}_{calc_period_name}": calc_period_def.total_period * prop}
            final_periods.update(new_period_dict)
            total_props += prop

        assert total_props == 1., f"Proportions for must sum to 1: {calc_period_name}"

    return final_periods
