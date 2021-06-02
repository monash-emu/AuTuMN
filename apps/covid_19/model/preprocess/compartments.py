from typing import Dict

from apps.covid_19.model.parameters import Sojourn


def calc_compartment_periods(sojourn: Sojourn) -> Dict[str, float]:
    """
    Calculate compartment periods from the provided splits into early and late components of the exposed and active
    periods.
    """

    final_periods = {**sojourn.compartment_periods}
    for calc_period_name, calc_period_def in sojourn.compartment_periods_calculated.items():
        period = calc_period_def.total_period
        props_def = calc_period_def.proportions
        total_props = 0
        for comp_name, prop in props_def.items():
            final_periods[f"{comp_name}_{calc_period_name}"] = period * prop
            total_props += prop

        assert total_props == 1, f"Proportions for {calc_period_name} must sum to 1"

    return final_periods
