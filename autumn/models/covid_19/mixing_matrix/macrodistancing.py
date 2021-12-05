from typing import Callable, Dict, List
import numpy as np
import pandas as pd

from autumn.models.covid_19.constants import BASE_DATETIME, LOCATIONS
from autumn.models.covid_19.parameters import Country, MixingLocation
from autumn.tools.curve import scale_up_function
from autumn.tools.inputs.mobility.queries import get_mobility_data
from autumn.tools.utils.utils import apply_moving_average


def weight_mobility_data(google_mob_df: pd.DataFrame, location_map: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Get weighted average for each modelled location from Google mobility estimates.
    Uses the user-specified request to define the mapping of Google mobility locations to the modelled locations.
    Modelled locations are:
        - work
        - other_locations
        - home
    Google mobility locations are:
        - retail_and_recreation
        - grocery_and_pharmacy
        - parks
        - transit_stations
        - residential
        - workplaces

    A standard example mapping (that we commonly use) would be:
    {
    'work':
        {'workplaces': 1.0},
    'other_locations':
        {'retail_and_recreation': 0.25, 'grocery_and_pharmacy': 0.25, 'parks': 0.25, 'transit_stations': 0.25},
    'home':
        {'residential': 1.0}
    }
    Values for the sub-dictionaries for each modelled location must sum to one
    (which is confirmed in parameter validation).

    If a location is not specified, that location will not be scaled - this will be sorted out in later stages of
    processing.

    Args:
        google_mob_df: The Google mobility data provided in raw form
        location_map: The mapping instructions from modelled locations (first keys) to Google locations (second keys)

    Returns:
        Dataframe containing the keys for scaling the modelled locations over time based on these mobility inputs

    """

    model_mob_df = pd.DataFrame(np.zeros((len(google_mob_df), len(location_map))), columns=list(location_map.keys()))
    for model_loc, google_locs in location_map.items():
        for g_loc in google_locs:
            model_mob_df[model_loc] += google_mob_df[g_loc] * location_map[model_loc][g_loc]

    return model_mob_df


def get_mobility_funcs(
        country: Country, region: str, mobility_requests: Dict[str, MixingLocation],
        google_mobility_locations: Dict[str, Dict[str, float]], square_mobility_effect: bool, smooth_google_data: bool,
) -> Dict[str, Callable[[float], float]]:
    """
    Loads Google mobility data, combines it with user requested timeseries data and then returns a mobility function for
    each location.

    Args:
        country: Country being simulated
        region: If a sub-region of the country is being simulated, this sub-region
        mobility_requests: The mixing location parameters
        google_mobility_locations: The mapping to model locations from Google mobility locations
        square_mobility_effect: See update_mixing_data
        smooth_google_data: Whether to smooth the raw Google mobility data for that location

    Returns:
        The final mobility functions for each modelled location

    """

    mob_df, google_mobility_days = get_mobility_data(country.iso3, region, BASE_DATETIME)
    model_loc_mobility_values = weight_mobility_data(mob_df, google_mobility_locations)

    # Currently the only options are to use raw mobility or 7-day moving average (although easy to change, of course)
    if smooth_google_data:
        for loc in model_loc_mobility_values.columns:
            model_loc_mobility_values[loc] = apply_moving_average(model_loc_mobility_values[loc], 7)

    # Build mixing data timeseries
    mobility_requests = {k: v.dict() for k, v in mobility_requests.items()}  # Needed as dict instead of parameters
    mobility_requests = update_mixing_data(mobility_requests, model_loc_mobility_values, google_mobility_days)

    # Build the time variant location-specific macrodistancing adjustment functions from mixing timeseries
    mobility_funcs = {}
    exponent = 2 if square_mobility_effect else 1
    for location, timeseries in mobility_requests.items():
        loc_vals = [v ** exponent for v in timeseries["values"]]
        mobility_funcs[location] = scale_up_function(timeseries["times"], loc_vals, method=4)

    return mobility_funcs


def update_mixing_data(
        mobility_requests: dict, google_mobility_values: pd.DataFrame, google_mobility_days: list
):
    """

    Args:
        mobility_requests: User requests mobility parameter object
        google_mobility_values: Dates for which we have Google mobility data available
        google_mobility_days: Values of Google mobility on these dates

    Returns:

    """

    last_google_day = google_mobility_days[-1]

    # Loop over all the modelled locations
    for loc_key in LOCATIONS:
        loc_mixing = mobility_requests.get(loc_key)

        # Check data lengths match
        if loc_mixing:
            msg = f"Mixing series length mismatch for {loc_key}"
            assert len(loc_mixing["times"]) == len(loc_mixing["values"]), msg

        # Add historical Google mobility data to user-specified mixing params
        if loc_key in google_mobility_values.columns:
            mobility_values = google_mobility_values[loc_key].to_list()

            # Just take the raw the mobility data for the location if no requests submitted
            if not loc_mixing:
                mobility_requests[loc_key] = {"times": google_mobility_days, "values": mobility_values}

            # Append user-specified mixing data to historical mobility data
            elif loc_mixing["append"]:
                first_append_day = min(loc_mixing["times"])
                if last_google_day < first_append_day:
                    # Appended days happen after the Google Mobility data, so we can just append them.
                    mobility_requests[loc_key]["times"] = google_mobility_days + loc_mixing["times"]
                    mobility_requests[loc_key]["values"] = mobility_values + loc_mixing["values"]

                # Appended days start during the Google mobility data, so we have to merge them
                else:
                    merge_idx = None
                    for idx, day in enumerate(google_mobility_days):
                        if day >= first_append_day:
                            merge_idx = idx
                            break
                    mobility_requests[loc_key]["times"] = google_mobility_days[: merge_idx] + loc_mixing["times"]
                    mobility_requests[loc_key]["values"] = mobility_values[: merge_idx] + loc_mixing["values"]

            # If no data have been loaded, no need to append, just insert the user-specified data
            else:
                mobility_requests[loc_key] = loc_mixing

        # No Google mobility data, but we still have user-specified data
        elif loc_mixing:
            msg = f"Cannot 'append' for {loc_key}: no Google mobility data available."
            assert not loc_mixing["append"], msg
            mobility_requests[loc_key] = loc_mixing

        # Interpret user requests if specified in a way other than absolute mobility values
        if loc_mixing:
            loc_mixing["values"] = parse_values(loc_mixing["values"])

    mobility_requests = {k: {"values": v["values"], "times": v["times"]} for k, v in mobility_requests.items()}
    return mobility_requests


def get_mobility_specific_period(
    country: str,
    region: str,
    google_mobility_locations: Dict[str, Dict[str, float]],
    split_dates: List[float],
) -> Dict[str, Callable[[float], float]]:
    """
    Loads google mobility data, splits it for the requested time duration
    and then returns a mobility function for each location.
    """

    google_mob_data, google_mobility_days = get_mobility_data(country, region, BASE_DATETIME)
    mobility_values = weight_mobility_data(google_mob_data, google_mobility_locations)

    first_timepoint_index = google_mobility_days.index(split_dates[0])
    second_timepoint_index = google_mobility_days.index(split_dates[1])

    split_google_mobility_days = google_mobility_days[first_timepoint_index: second_timepoint_index]
    spilt_google_mobility_values = {
        loc: mobility_values.loc[first_timepoint_index: second_timepoint_index - 1, loc].to_list() for
        loc in mobility_values.columns
    }
    return split_google_mobility_days, spilt_google_mobility_values


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
            func = PARSE_FUNCS[func_name]
            new_val = func(*args)
        else:
            # Do not change.
            new_val = v

        new_values.append(new_val)

    return new_values


def repeat_prev(prev_vals: List[float]):
    """
    Repeats the previous seen value again
    """
    return prev_vals[-1]


def add_to_prev(prev_vals: List[float], increment: float):
    """
    Add increment to previous
    """
    val = prev_vals[-1] + increment
    if val < 0:
        return 0
    else:
        return val


def add_to_prev_up_to_1(prev_vals: List[float], increment: float):
    """
    Add increment to previous
    """
    val = prev_vals[-1] + increment
    if val > 1:
        return 1
    elif val < 0:
        return 0
    else:
        return val


def scale_prev(prev_vals: List[float], fraction: float):
    """
    Apply a percentage to the previous value, saturating at zero
    """
    val = prev_vals[-1] * fraction
    if val < 0:
        return 0
    else:
        return val


def scale_prev_up_to_1(prev_vals: List[float], fraction: float):
    """
    Apply a percentage to the previous value, saturating at one or zero
    """
    val = prev_vals[-1] * fraction
    if val > 1:
        return 1
    elif val < 0:
        return 0
    else:
        return val


def close_gap_to_1(prev_vals: List[float], fraction: float):
    prev_val = prev_vals[-1]
    return (1.0 - prev_val) * fraction + prev_val


def max_last_period(prev_vals: List[float], period: int):
    last_n_values = min(len(prev_vals), period)
    return max(prev_vals[-last_n_values:])


def min_last_period(prev_vals: List[float], period: int):
    last_n_values = min(len(prev_vals), period)
    return min(prev_vals[-last_n_values:])


def average_mobility(prev_vals: List[float], period: int):
    last_n_values = min(len(prev_vals), period)
    return sum(prev_vals[-last_n_values:])/len(prev_vals[-last_n_values:])


def copy_mobility(prev_vals: List[float], ignore_vals: int):
    """
    returns the mobility level at the requested time by ignoring the last values defined by ignore_vals
    """
    ignore_vals = ignore_vals+1
    prev_val = (prev_vals[-ignore_vals:])
    return prev_val[1]


def close_to_max_last_period(prev_vals: List[float], period: int, fraction: float):
    """
    Partial return from last mobility estimate to the highest level observed over the recent period specified.
    """

    last_n_values = min(len(prev_vals), period)
    max_val_last_period = max(prev_vals[-last_n_values:])
    prev_val = prev_vals[-1]
    return (max_val_last_period - prev_val) * fraction + prev_val


# Used for the Philippines
CQ_MOBILITY = {
    # GCQ Reference period: from May 15 2021
    "GCQ": {
        "work": .70,
        "other_locations": .70
    },
    # MECQ Reference period April 12-May 15 2021
    "MECQ": {
        "work": .55,
        "other_locations": .55
    },
    # ECQ Reference period March 29-April 11 2021
    "ECQ": {
        "work": .40,
        "other_locations": .40
    },
}


def ecq(prev_vals: List[float], loc_key: str):
    return CQ_MOBILITY["ECQ"][loc_key]


def mecq(prev_vals: List[float], loc_key: str):
    return CQ_MOBILITY["MECQ"][loc_key]


def gcq(prev_vals: List[float], loc_key: str):
    return CQ_MOBILITY["GCQ"][loc_key]


PARSE_FUNCS = {
    "repeat_prev": repeat_prev,
    "add_to_prev": add_to_prev,
    "add_to_prev_up_to_1": add_to_prev_up_to_1,
    "scale_prev": scale_prev,
    "scale_prev_up_to_1": scale_prev_up_to_1,
    "close_gap_to_1": close_gap_to_1,
    "max_last_period": max_last_period,
    "close_to_max_last_period": close_to_max_last_period,
    "min_last_period": min_last_period,
    "copy_mobility": copy_mobility,
    "average_mobility":average_mobility,
    # used for the Philippines
    "ECQ": ecq,
    "MECQ": mecq,
    "GCQ": gcq,
}
