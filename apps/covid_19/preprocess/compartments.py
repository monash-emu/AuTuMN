from typing import Dict


def calc_compartment_periods(
    periods: Dict[str, float], periods_calculated: dict
) -> Dict[str, float]:
    """
    Calculate compartment periods from provided parameters.
    """
    final_periods = {**periods}
    for calc_period_name, calc_period_def in periods_calculated.items():
        period = calc_period_def["total_period"]
        props_def = calc_period_def["proportions"]
        total_props = 0
        for comp_name, prop in props_def.items():
            final_periods[comp_name] = period * prop
            total_props += prop

        assert total_props == 1, f"Proportions for {calc_period_name} must sum to 1"

    return final_periods
