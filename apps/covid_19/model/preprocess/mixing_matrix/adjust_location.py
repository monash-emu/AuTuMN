from typing import Callable, Dict

import numpy as np

from autumn.curve import scale_up_function, tanh_based_scaleup
from autumn.inputs import get_country_mixing_matrix, get_mobility_data
from autumn.tool_kit.utils import apply_moving_average
from apps.covid_19.model.parameters import (
    Country,
    Mobility,
    MixingLocation,
)

from .adjust_base import BaseMixingAdjustment
from apps.covid_19.constants import BASE_DATE, BASE_DATETIME
from apps.covid_19.model.preprocess.mixing_matrix import utils

from . import funcs as parse_funcs

# Locations that can be used for mixing
LOCATIONS = ["home", "other_locations", "school", "work"]


class LocationMixingAdjustment(BaseMixingAdjustment):
    def __init__(self, country: Country, mobility: Mobility):
        """Build the time variant location adjustment functions"""
        country_iso3 = country.iso3
        region = mobility.region
        mixing = mobility.mixing
        npi_effectiveness_params = mobility.npi_effectiveness
        google_mobility_locations = mobility.google_mobility_locations
        microdistancing_params = mobility.microdistancing
        smooth_google_data = mobility.smooth_google_data

        # Load mobility data
        google_mobility_values, google_mobility_days = get_mobility_data(
            country_iso3, region, BASE_DATETIME, google_mobility_locations
        )
        if smooth_google_data:
            for loc in google_mobility_values:
                google_mobility_values[loc] = apply_moving_average(google_mobility_values[loc], 7)

        # Build mixing data timeseries
        mixing = update_mixing_data(
            {k: v.dict() for k, v in mixing.items()},
            npi_effectiveness_params,
            google_mobility_values,
            google_mobility_days,
        )

        # Build the time variant location-specific macrodistancing adjustment functions from mixing timeseries
        macrodistancing_locations = [loc for loc in LOCATIONS if loc in mixing]
        self.loc_adj_funcs = {}
        for loc_key in macrodistancing_locations:
            loc_times = mixing[loc_key]["times"]
            loc_vals = mixing[loc_key]["values"]
            self.loc_adj_funcs[loc_key] = \
                scale_up_function(loc_times, loc_vals, method=4)

        # Apply the microdistancing function
        self.microdistancing_function = \
            apply_microdistancing(microdistancing_params) if \
                microdistancing_params else \
                None

        # Load all location-specific mixing info.
        self.matrix_components = {}
        for sheet_type in ["all_locations"] + LOCATIONS:
            # Loads a 16x16 ndarray
            self.matrix_components[sheet_type] = get_country_mixing_matrix(sheet_type, country_iso3)

    def get_adjustment(
        self, time: float, mixing_matrix: np.ndarray, microdistancing_locations: list
    ) -> np.ndarray:
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
            if self.microdistancing_function and loc_key in microdistancing_locations:
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
):
    most_recent_day = google_mobility_days[-1]
    for loc_key in LOCATIONS:
        loc_mixing = mixing.get(loc_key)
        if loc_mixing:
            # Ensure this location's mixing times and values match.
            assert len(loc_mixing["times"]) == len(
                loc_mixing["values"]
            ), f"Mixing series length mismatch for {loc_key}"

        # Add historical Google mobility data to user-specified mixing params
        mobility_values = google_mobility_values.get(loc_key)
        if mobility_values:
            # Google mobility values for this location
            if not loc_mixing:
                # Just insert the mobility data
                mixing[loc_key] = {
                    "times": google_mobility_days,
                    "values": mobility_values,
                }
            elif loc_mixing["append"]:
                # Append user-specified mixing data to historical mobility data
                first_append_day = min(loc_mixing["times"])
                if most_recent_day < first_append_day:
                    # Appended days happen after the Google Mobility data, so we can just append them.
                    mixing[loc_key]["times"] = google_mobility_days + loc_mixing["times"]
                    mixing[loc_key]["values"] = mobility_values + loc_mixing["values"]
                else:
                    # Appended days start during the Google Mobility data, so we must merge them.
                    merge_idx = None
                    for idx, day in enumerate(google_mobility_days):
                        if day >= first_append_day:
                            merge_idx = idx
                            break
                    mixing[loc_key]["times"] = (
                        google_mobility_days[:merge_idx] + loc_mixing["times"]
                    )
                    mixing[loc_key]["values"] = mobility_values[:merge_idx] + loc_mixing["values"]

            else:
                # Don't append, overwrite: insert the user-specified data
                mixing[loc_key]["times"] = loc_mixing["times"]
                mixing[loc_key]["values"] = loc_mixing["values"]
        elif loc_mixing:
            # No Google mobility data, but we still have user-specified data.
            if not loc_mixing["append"]:
                # Use user-specified data
                mixing[loc_key]["times"] = loc_mixing["times"]
                mixing[loc_key]["values"] = loc_mixing["values"]
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

    mixing = {k: {"values": v["values"], "times": v["times"]} for k, v in mixing.items()}
    return mixing


def apply_microdistancing(params):
    """
    Work out microdistancing function to be applied as a multiplier to some or all of the Prem locations as requested in
    the microdistancing_locations.
    """

    # Collate the components to the microdistancing function
    microdist_component_funcs = []
    for microdist_type in params:
        if params[microdist_type].function_type == "tanh":
            microdist_component_funcs.append(tanh_based_scaleup(**params[microdist_type].parameters.dict()))
        elif params[microdist_type].function_type == "empiric":
            micro_times = params[microdist_type].parameters.times
            multiplier = params[microdist_type].parameters.max_effect
            micro_vals = [
                1.0 - multiplier * value for value in params[microdist_type].parameters.values
            ]
            microdist_component_funcs.append(scale_up_function(micro_times, micro_vals, method=4))

    # Generate the overall composite function representing the product of all the components
    def microdist_composite_func(time):
        return \
            np.product(
                [
                    microdist_component_funcs[i_func](time) for
                    i_func in range(len(microdist_component_funcs))
                ]
            )

    return microdist_composite_func


def parse_values(values):
    """
    Convert all mixing time series values to a float
    """
    new_values = []
    for v in values:
        if type(v) is list:
            # Apply a function to the history to get the next value.
            func_name = v[0]
            args = [new_values] + v[1:]
            func = getattr(parse_funcs, func_name)
            new_val = func(*args)
        else:
            # Do not change.
            new_val = v

        new_values.append(new_val)

    return new_values
