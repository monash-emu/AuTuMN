from typing import Dict

from autumn.models.covid_19.parameters import Sojourn


def calc_compartment_periods(sojourn: Sojourn) -> Dict[str, float]:
    """
    Calculate compartment periods from the provided splits into early and late components of the exposed and active
    periods.
    """

    final_periods = {**sojourn.compartment_periods}
    for calc_period_name, calc_period_def in sojourn.compartment_periods_calculated.items():
        comp_final_periods = {
            f"{comp_name}_{calc_period_name}": calc_period_def.total_period * prop for
            comp_name, prop in calc_period_def.proportions.items()
        }
        final_periods.update(comp_final_periods)

    return final_periods
