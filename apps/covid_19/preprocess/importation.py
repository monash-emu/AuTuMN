from typing import List
from autumn.curve import scale_up_function


def get_importation_rate_func_as_birth_rates(
    importation_times: List[float],
    importation_n_cases: List[float],
    detect_prop_func,
    starting_pops: list,
):
    """
    When imported cases are explicitly simulated as part of the modelled population. They enter the late_infectious
    compartment through a birth process.
    """

    # Inflate importation numbers to account for undetected cases (assumed to be asymptomatic or sympt non hospital)
    actual_imported_cases = \
        [importation_n_cases[i_time] / detect_prop_func(time) for
         i_time, time in enumerate(importation_times)]

    # Scale-up curve for importation numbers
    importation_numbers_scale_up = scale_up_function(
        importation_times, actual_imported_cases, method=4, smoothness=5.0, bound_low=0.0
    )

    # Divide through by total population to create per capita rate and return
    def recruitment_rate(t):
        return importation_numbers_scale_up(t) / sum(starting_pops)

    return recruitment_rate


# dummy proportions for now:
# FIXME: These are parameters!
IMPORTATION_PROPS_BY_AGE = {
    "0": 0.04,
    "5": 0.04,
    "10": 0.04,
    "15": 0.04,
    "20": 0.08,
    "25": 0.09,
    "30": 0.09,
    "35": 0.09,
    "40": 0.09,
    "45": 0.08,
    "50": 0.08,
    "55": 0.08,
    "60": 0.04,
    "65": 0.04,
    "70": 0.04,
    "75": 0.04,
}
