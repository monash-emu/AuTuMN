from typing import List, Callable
from autumn.curve import scale_up_function


def get_importation_rate_func(
    country: str,
    importation_times: List[float],
    importation_n_cases: List[float],
    self_isolation_effect: float,
    enforced_isolation_effect: float,
    contact_rate: float,
    starting_population: float,
) -> Callable[[float], float]:
    """
    Returns a time varying function of importation secondary infection rate.
    See also: flows.

    Used when imported cases need to be accounted for to inflate the force of infection but they are not explicitly included
    in the modelled population.
    """

    # Number of importation over time
    get_importation_amount = scale_up_function(importation_times, importation_n_cases)

    # time-variant infectiousness of imported cases
    assert country == "victoria", "VIC only. Hard-coded Victorian values."
    mystery_times = [75.0, 77.0, 88.0, 90.0]
    mystery_vals = [
        1.0,
        1.0 - self_isolation_effect,
        1.0 - self_isolation_effect,
        1.0 - enforced_isolation_effect,
    ]
    tv_imported_infectiousness = scale_up_function(mystery_times, mystery_vals, method=4,)

    def recruitment_rate(t):
        return (
            get_importation_amount(t)
            * tv_imported_infectiousness(t)
            * contact_rate
            / starting_population
        )

    return recruitment_rate


def get_importation_rate_func_as_birth_rates(
    importation_times: List[float],
    importation_n_cases: List[float],
    detect_prop_func,
    starting_population: float,
):
    """
    When imported cases are explicitly simulated as part of the modelled population. They enter the late_infectious
    compartment through a birth process
    """
    # inflate importation numbers to account for undetected cases (assumed to be asymptomatic or sympt non hospital)
    for i, time in enumerate(importation_times):
        importation_n_cases[i] /= detect_prop_func(time)
    # scale-up curve for importation numbers
    importation_numbers_scale_up = scale_up_function(
        importation_times, importation_n_cases, method=5, smoothness=5.0, bound_low=0.0
    )

    def recruitment_rate(t):
        return importation_numbers_scale_up(t) / starting_population

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
