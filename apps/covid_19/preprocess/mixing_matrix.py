from datetime import date, datetime
from typing import Callable

import numpy as np

from autumn.curve import scale_up_function

from autumn.demography.social_mixing import load_specific_prem_sheet

# Base date used to calculate mixing matrix times.
BASE_DATE = date(2010, 12, 31)

# Locations that can be used for mixing
LOCATIONS = ["home", "other_locations", "school", "work"]
AGE_INDICES = list(range(16))


def build_static(country: str, multipliers: np.ndarray) -> np.ndarray:
    """
    Get a non-time-varying mixing matrix.
    multipliers is a matrix with the ages-specific multipliers.
    Returns the updated mixing-matrix
    """
    mixing_matrix = load_specific_prem_sheet("all_locations", country)
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
    is_reinstall_regular_prayers: bool,
    prayers_params: dict,
    end_time: float,
) -> Callable[[float], dict]:
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
        matrix_components[sheet_type] = load_specific_prem_sheet(sheet_type, country)

    # Update the mixing parameters to simulate re-installing regular Friday prayers from t_start to t_end. We assume that
    # a proportion 'prop_participating' of the population participates in the prayers and that the other-location
    # contact rates are multiplied by 'other_location_multiplier' for the participating individuals.
    if is_reinstall_regular_prayers:
        t_start = prayers_params["restart_time"]
        t_end = end_time
        prop_participating = prayers_params["prop_participating"]
        contact_multiplier = prayers_params["contact_multiplier"]
        other_locations = mixing.get("other_locations")
        assert other_locations, "need to specify other_location mixing params"
        t = t_start
        reference_val = other_locations["values"][-1]
        amplified_val = reference_val * (
            (1.0 - prop_participating) + contact_multiplier * prop_participating
        )
        while t < t_end:
            other_locations["times"] += [t, t + 1, t + 2]
            other_locations["values"] += [reference_val, amplified_val, reference_val]
            t += 7

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
        complement_indices = [i for i in AGE_INDICES if i not in affected_age_indices]

        for age_idx_affected in affected_age_indices:
            age_idx_key = f"age_{age_idx_affected}"
            age_times = mixing[age_idx_key]["times"]
            age_vals = mixing[age_idx_key]["values"]
            age_adj_func = scale_up_function(loc_times, loc_vals, method=4,)
            age_adj_val = age_adj_func(time)
            for age_idx_not_affected in complement_indices:
                mixing_matrix[age_idx_affected, age_idx_not_affected] *= age_adj_val
                mixing_matrix[age_idx_not_affected, age_idx_affected] *= age_adj_val

            # FIXME: patch for elderly cocooning in Victoria assuming
            # FIXME: ... assuming what?
            for idx in affected_age_indices:
                mixing_matrix[age_idx_affected, idx] *= 1.0 - (1.0 - age_adj_val) / 2.0

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
