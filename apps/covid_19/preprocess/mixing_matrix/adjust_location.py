from typing import Callable

import numpy as np

from autumn.curve import scale_up_function, tanh_based_scaleup

from autumn.inputs import get_country_mixing_matrix, get_mobility_data

from .adjust_base import BaseMixingAdjustment
from .utils import BASE_DATE, BASE_DATETIME

# Locations that can be used for mixing
LOCATIONS = ["home", "other_locations", "school", "work"]
MICRODISTANCING_LOCATIONS = ["school", "other_locations", "work"]


class LocationMixingAdjustment(BaseMixingAdjustment):
    def __init__(
        self,
        country_iso3: str,
        region: str,
        mixing: dict,
        npi_effectiveness_params: dict,
        google_mobility_locations: dict,
        is_periodic_intervention: bool,
        periodic_int_params: dict,
        periodic_end_time: float,
        microdistancing_params: dict,
    ):
        """Build the time variant location adjustment functions"""
        # Load mobility data
        google_mobility_values, google_mobility_days = get_mobility_data(
            country_iso3, region, BASE_DATETIME, google_mobility_locations
        )
        # Build mixing data timeseries
        mixing = update_mixing_data(
            mixing,
            npi_effectiveness_params,
            google_mobility_values,
            google_mobility_days,
            is_periodic_intervention,
            periodic_int_params,
            periodic_end_time,
        )
        # Build the time variant location adjustment functions from mixing timeseries
        mixing_locations = [loc for loc in LOCATIONS if loc in mixing]
        self.loc_adj_funcs = {}
        for loc_key in mixing_locations:
            loc_times = mixing[loc_key]["times"]
            loc_vals = mixing[loc_key]["values"]
            self.loc_adj_funcs[loc_key] = scale_up_function(loc_times, loc_vals, method=4)

        # Work out microdistancing function to be applied to all non-household locations
        self.microdistancing_function = None
        if microdistancing_params:
            self.microdistancing_function = tanh_based_scaleup(**microdistancing_params)

        # Load all location-specific mixing info.
        self.matrix_components = {}
        for sheet_type in ["all_locations"] + LOCATIONS:
            # Loads a 16x16 ndarray
            self.matrix_components[sheet_type] = get_country_mixing_matrix(sheet_type, country_iso3)

    def get_adjustment(self, time: float, mixing_matrix: np.ndarray) -> np.ndarray:
        """
        Apply time-varying location adjustments.
        Returns a new mixing matrix, modified to adjust for dynamic mixing changes for a given point in time.
        """
        for loc_key in LOCATIONS:

            # Start the adjustment value for each location from a value of 1 for "no adjustment".
            loc_adjustment = 1

            # Adjust for Google Mobility data.
            if loc_key in self.loc_adj_funcs:
                loc_adj_func = self.loc_adj_funcs[loc_key]
                loc_adjustment *= loc_adj_func(time)

            # Adjust for microdistancing
            if self.microdistancing_function and loc_key in MICRODISTANCING_LOCATIONS:
                loc_adjustment *= self.microdistancing_function(time)

            # Apply adjustment by subtracting the contacts that need to come off
            loc_adjustment_matrix = (loc_adjustment - 1) * self.matrix_components[loc_key]
            mixing_matrix = np.add(mixing_matrix, loc_adjustment_matrix)

        return mixing_matrix


def update_mixing_data(
    mixing: dict,
    npi_effectiveness_params: dict,
    google_mobility_values: dict,
    google_mobility_days: list,
    is_periodic_intervention: bool,
    periodic_int_params: dict,
    periodic_end_time: float,
):
    most_recent_day = google_mobility_days[-1]
    for loc_key in LOCATIONS:

        loc_mixing = mixing.get(loc_key)
        if loc_mixing:
            assert len(loc_mixing["times"]) == len(
                loc_mixing["values"]
            ), f"Mixing series length mismatch for {loc_key}"
            loc_mixing["times"] = [
                (time_date - BASE_DATE).days for time_date in loc_mixing["times"]
            ]

            # Ensure all user-specified dynamic mixing is up to date
            is_fresh_timeseries = max(loc_mixing["times"]) >= most_recent_day
            assert (
                is_fresh_timeseries
            ), f"Dynamic mixing for {loc_key} is out of date, max date less than {most_recent_day}"

        # Add historical Google mobility data to user-specified mixing params
        mobility_values = google_mobility_values.get(loc_key)
        if mobility_values:
            # Google moblity values for this location
            if not loc_mixing:
                # Just insert the mobility data
                mixing[loc_key] = {
                    "times": google_mobility_days,
                    "values": mobility_values,
                }
            elif loc_mixing["append"]:
                # Append user-specified mixing data to historical mobility data
                first_append_day = min(loc_mixing["times"])
                assert most_recent_day < first_append_day, f"Cannot append {loc_key}, dates clash."
                mixing[loc_key] = {
                    "times": google_mobility_days + loc_mixing["times"],
                    "values": mobility_values + loc_mixing["values"],
                }
            else:
                # Don't append, overwrite: insert the user-specified data
                mixing[loc_key] = {
                    "times": loc_mixing["times"],
                    "values": loc_mixing["values"],
                }
        elif loc_mixing:
            # No Google mobility data, but we still have user-specified data.
            if not loc_mixing["append"]:
                # Use user-specified data
                mixing[loc_key] = {
                    "times": loc_mixing["times"],
                    "values": loc_mixing["values"],
                }
            else:
                # User has specified "append", but there is nothing to append to.
                msg = f"Cannot 'append' for {loc_key}: no Google mobility data available."
                raise ValueError(msg)

        # Convert % adjustments to fractions
        loc_mixing = mixing.get(loc_key)
        if loc_mixing:
            loc_mixing["values"] = parse_values(loc_mixing["values"])

        # Adjust the mixing parameters by scaling them according to NPI effectiveness
        npi_adjust_val = npi_effectiveness_params.get(loc_key)
        if npi_adjust_val:
            mixing[loc_key]["values"] = [
                1 - (1 - val) * npi_adjust_val for val in mixing[loc_key]["values"]
            ]

    # Update the mixing parameters to simulate a future regular periodic process
    if is_periodic_intervention:
        mixing["other_locations"] = add_periodic_intervention(
            periodic_int_params, mixing["other_locations"], periodic_end_time
        )
    return mixing


def parse_values(values):
    """
    Convert all mixing time series values to a float
    """
    new_values = []
    prev = None
    for v in values:
        if type(v) is str and prev and v.endswith("%"):
            # Make this value a percent of the prev value.
            fraction = float(v.replace("%", "")) / 100
            new_val = prev * fraction
        else:
            # Do not change.
            new_val = v

        new_values.append(new_val)
        prev = new_val

    return new_values


def add_periodic_intervention(periodic_int_params, old_other_locations, end_time):
    """
    We assume that a proportion 'prop_participating' of the population participates in the intervention and that the
    other-location contact rates are multiplied by 'other_location_multiplier' for the participating individuals.
    """
    # Make a copy of the data
    other_locations = {
        "values": list(old_other_locations["values"]),
        "times": list(old_other_locations["times"]),
    }
    # Avoid over-writing existing times, find start and end time
    t_start = max([periodic_int_params["restart_time"], max(other_locations["times"]) + 1])
    t = t_start
    t_end = end_time

    # Extract parameters
    prop_participating, contact_multiplier, duration, period = (
        periodic_int_params["prop_participating"],
        periodic_int_params["contact_multiplier"],
        periodic_int_params["duration"],
        periodic_int_params["period"],
    )
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
