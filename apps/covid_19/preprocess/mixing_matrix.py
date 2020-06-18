from datetime import date, datetime
from typing import Callable

import numpy as np

from autumn.curve import scale_up_function

from autumn.demography.social_mixing import load_country_mixing_matrix

# Base date used to calculate mixing matrix times.
BASE_DATE = date(2019, 12, 31)

# Locations that can be used for mixing
LOCATIONS = ["home", "other_locations", "school", "work"]
AGE_INDICES = list(range(16))


def add_periodic_intervention(periodic_int_params, other_locations, end_time):
    """
    We assume that a proportion 'prop_participating' of the population participates in the intervention and that the
    other-location contact rates are multiplied by 'other_location_multiplier' for the participating individuals.
    """

    # Avoid over-writing existing times, find start and end time
    t_start = max([periodic_int_params["restart_time"], max(other_locations["times"]) + 1])
    t = t_start
    t_end = end_time

    # Extract parameters
    prop_participating, contact_multiplier, duration, period = \
        periodic_int_params["prop_participating"], periodic_int_params["contact_multiplier"], \
        periodic_int_params["duration"], periodic_int_params["period"]
    reference_val = other_locations["values"][-1]

    # Calculate the value for other locations that the contact rate increases to
    amplified_val = reference_val * (
            (1.0 - prop_participating) + contact_multiplier * prop_participating
    )

    # Extend dictionary of other locations
    while t < t_end:
        other_locations["times"] += [t, t + 1, t + 1 + duration]
        other_locations["values"] += [reference_val, amplified_val, reference_val]
        t += period

    return other_locations


def build_static(country: str, multipliers: np.ndarray) -> np.ndarray:
    """
    Get a non-time-varying mixing matrix.
    multipliers is a matrix with the ages-specific multipliers.
    Returns the updated mixing-matrix
    """
    mixing_matrix = load_country_mixing_matrix("all_locations", country)
    if multipliers:
        # Update the mixing matrix using some age-specific multipliers
        assert mixing_matrix.shape == multipliers.shape
        return np.multiply(mixing_matrix, multipliers)
    else:
        return mixing_matrix


def build_dynamic(
    country: str,
    mixing_params: dict,
    npi_effectiveness_params: dict,
    is_periodic_intervention: bool,
    periodic_int_params: dict,
    end_time: float,
) -> Callable[[float], np.ndarray]:
    """
    Build a time-varing mixing matrix
    """
    # Transform mixing params into structured format.
    base_keys = set()
    for k in mixing_params.keys():
        base_keys.add(k.replace("_times", "").replace("_values", ""))

    restructured_mixing_params = {}
    for k in base_keys:
        restructured_mixing_params[k] = {
            "times": mixing_params[f"{k}_times"],
            "values": mixing_params[f"{k}_values"],
        }

    # Preprocess mixing instructions for all included locations
    mixing = {}
    for location_key in restructured_mixing_params.keys():
        mixing_data = restructured_mixing_params[location_key]
        mixing[location_key] = {
            "values": mixing_data["values"],
            "times": list(parse_times(mixing_data["times"])),
        }

    # Adjust the mixing parameters according by scaling them according to NPI effectiveness
    for location_key, adjustment_val in npi_effectiveness_params.items():
        mixing[location_key]["values"] = [
            1 - (1 - val) * adjustment_val for val in mixing[location_key]["values"]
        ]

    # Load all location-specific mixing info.
    matrix_components = {}
    for sheet_type in ["all_locations"] + LOCATIONS:
        # Loads a 16x16 ndarray
        matrix_components[sheet_type] = load_country_mixing_matrix(sheet_type, country)

    # Update the mixing parameters to simulate a future regular periodic process
    if is_periodic_intervention:
        other_locations = mixing.get("other_locations")
        assert other_locations, "need to specify other_location mixing params"
        mixing["other_locations"] = add_periodic_intervention(periodic_int_params, mixing, end_time)

    def mixing_matrix_function(time: float):
        mixing_matrix = matrix_components["all_locations"]

        # Make adjustments by location
        for loc_key in [loc for loc in LOCATIONS if loc in mixing]:
            loc_times = mixing[loc_key]["times"]
            loc_vals = mixing[loc_key]["values"]
            loc_adj_func = scale_up_function(loc_times, loc_vals, method=4)
            location_adjustment_matrix = (loc_adj_func(time) - 1.0) * matrix_components[loc_key]
            mixing_matrix = np.add(mixing_matrix, location_adjustment_matrix)

        # Make adjustments by age
        affected_age_indices = [i for i in AGE_INDICES if f"age_{i}" in mixing]
        age_adjustment_functions = {}
        for age_idx_affected in affected_age_indices:
            age_idx_key = f"age_{age_idx_affected}"
            age_times = mixing[age_idx_key]["times"]
            age_vals = mixing[age_idx_key]["values"]
            age_adjustment_functions[age_idx_affected] = scale_up_function(
                age_times, age_vals, method=4,
            )

        for row_index in range(len(AGE_INDICES)):
            row_multiplier = (
                age_adjustment_functions[row_index](time)
                if row_index in affected_age_indices
                else 1.0
            )
            for col_index in range(len(AGE_INDICES)):
                col_multiplier = (
                    age_adjustment_functions[col_index](time)
                    if col_index in affected_age_indices
                    else 1.0
                )
                mixing_matrix[row_index, col_index] *= row_multiplier * col_multiplier

        return mixing_matrix

    return mixing_matrix_function


def parse_times(times):
    """
    Ensure all times are an integer,
    representing days since simulation start.
    """
    for time in times:
        if type(time) is str:
            time_date = datetime.strptime(time, "%Y%m%d").date()
            yield (time_date - BASE_DATE).days
        else:
            yield time


def get_total_contact_rates_by_age(mixing_matrix, direction="horizontal"):
    """
    Sum the contact-rates by age group
    :param mixing_matrix: the input mixing matrix
    :param direction: either 'horizontal' (infectee's perspective) or 'vertical' (infector's perspective)
    :return: dict
        keys are the age categories and values are the aggregated contact rates
    """
    assert direction in [
        "horizontal",
        "vertical",
    ], "direction should be in ['horizontal', 'vertical']"
    aggregated_contact_rates = {}
    for i in range(16):
        if direction == "horizontal":
            aggregated_contact_rates[str(5 * i)] = mixing_matrix[i, :].sum()
        else:
            aggregated_contact_rates[str(5 * i)] = mixing_matrix[:, i].sum()
    return aggregated_contact_rates
