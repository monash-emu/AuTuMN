from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from autumn.models.sm_jax.constants import LOCATIONS
from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.models.sm_jax.parameters import Country, MixingLocation
from autumn.core.inputs.mobility.queries import get_mobility_data
from autumn.core.utils.utils import apply_moving_average


def weight_mobility_data(
    google_mob_df: pd.DataFrame, location_map: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
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

    model_mob_df = pd.DataFrame(
        np.zeros((len(google_mob_df), len(location_map))),
        columns=list(location_map.keys()),
    )
    for model_loc, google_locs in location_map.items():
        for g_loc in google_locs:
            if not all(google_mob_df[g_loc].isna()):
                model_mob_df[model_loc] += google_mob_df[g_loc] * location_map[model_loc][g_loc]

    return model_mob_df


def get_mobility_funcs(
    country: Country,
    region: str,
    additional_mobility: Dict[str, Tuple[np.ndarray, np.ndarray]],
    google_mobility_locations: Dict[str, Dict[str, float]],
    square_mobility_effect: bool,
    smooth_google_data: bool,
    random_process_func,
    hh_contact_increase
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
        random_process_func: Random process function
        hh_contact_increase: relative household contact increase when schools are closed
    Returns:
        The final mobility functions for each modelled location

    """

    
    mob_df, google_mobility_days = get_mobility_data(country.iso3, region, COVID_BASE_DATETIME)
    model_loc_mobility_values = weight_mobility_data(mob_df, google_mobility_locations)

    # Currently the only options are to use raw mobility or 7-day moving average (although easy to change, of course)
    if smooth_google_data:
        for loc in model_loc_mobility_values.columns:
            model_loc_mobility_values[loc] = apply_moving_average(model_loc_mobility_values[loc], 7)

    # Build mixing data timeseries (change to dict, rather than parameters object)
    #mobility_requests = {}#{k: v.dict() for k, v in mobility_requests.items()}
    
    # OK, so this is now just a glorified and irritating way to convert our lists to pandas series
    # We don't do anything to our additional series - these are already correct, we're just converting
    # the google series we obtained above
    google_mobility_series = update_mixing_data(
        {}, model_loc_mobility_values, google_mobility_days
    )

    from summer2.functions.time import get_sigmoidal_interpolation_function
    from summer2.functions.util import capture_dict

    # Build the time variant location-specific macrodistancing adjustment functions from mixing timeseries
    mobility_funcs = {}
    for location, timeseries in google_mobility_series.items():
        if square_mobility_effect:
            timeseries = timeseries * timeseries
        mobility_funcs[location] = get_sigmoidal_interpolation_function(
            timeseries.index, timeseries
        )

    for location, (idx, timeseries) in additional_mobility.items():
        if square_mobility_effect:
            timeseries = timeseries * timeseries
        mobility_funcs[location] = get_sigmoidal_interpolation_function(
            idx, timeseries
        )

    # Increase household contacts when schools are closed
    if "school" in additional_mobility:
        (idx, timeseries) = additional_mobility["school"]
        prop_at_school = get_sigmoidal_interpolation_function(idx, timeseries)
        hh_adjustment = 1. + (1. - prop_at_school) * hh_contact_increase
        if "home" in mobility_funcs:
            mobility_funcs["home"] = mobility_funcs["home"] * hh_adjustment
        else:
            mobility_funcs["home"] = hh_adjustment

    if random_process_func:
        rp_adjustment = random_process_func * random_process_func if square_mobility_effect else random_process_func
        for location in ["work", "other_locations"]:
            mobility_funcs[location] = mobility_funcs[location] * rp_adjustment

    return capture_dict(**mobility_funcs)


def update_mixing_data(
    mob_values: dict, google_mobility_values: pd.DataFrame, google_mobility_days: list
) -> Dict[str, Dict[str, float]]:
    """
    Incorporate the user requests relating to mobility change with the Google mobility data, according to how the
    requests have been submitted.

    Args:
        mob_values: User requests mobility parameter object
        google_mobility_values: Dates for which we have Google mobility data available
        google_mobility_days: Values of Google mobility on these dates

    Returns:
        The final processed mobility values by location with standard keys for each location (times and values)

    """

    # Loop over all the modelled locations
    for loc_key in LOCATIONS:
        loc_mixing = mob_values.get(loc_key)

        # Check data lengths match
        if loc_mixing:
            msg = f"Mixing series length mismatch for {loc_key}"
            assert len(loc_mixing["times"]) == len(loc_mixing["values"]), msg

        # Add historical Google mobility data to user-specified mixing params
        if loc_key in google_mobility_values.columns:
            mobility_values = google_mobility_values[loc_key].to_list()

            # Just take the raw the mobility data for the location if no requests submitted
            if not loc_mixing:
                mob_values[loc_key] = {
                    "times": google_mobility_days,
                    "values": mobility_values,
                }

            # Append user-specified mixing data to historical mobility data
            elif loc_mixing["append"]:
                first_append_day = min(loc_mixing["times"])

                # All requested dates are after the Google Mobility data starts, so we can just join the two lists
                if google_mobility_days[-1] < first_append_day:
                    mob_values[loc_key]["times"] = google_mobility_days + loc_mixing["times"]
                    mob_values[loc_key]["values"] = mobility_values + loc_mixing["values"]

                # Requested days start during the Google mobility data, so we truncate the Google mobility data
                else:
                    merge_idx = None
                    for idx, day in enumerate(google_mobility_days):
                        if day >= first_append_day:
                            merge_idx = idx
                            break
                    mob_values[loc_key]["times"] = (
                        google_mobility_days[:merge_idx] + loc_mixing["times"]
                    )
                    mob_values[loc_key]["values"] = (
                        mobility_values[:merge_idx] + loc_mixing["values"]
                    )

            # If no data have been loaded, no need to append, just use the user-specified requests directly
            else:
                mob_values[loc_key] = loc_mixing

        # No Google mobility data, but we still have user-specified data
        elif loc_mixing:
            msg = f"Cannot 'append' for {loc_key}: no Google mobility data available."
            assert not loc_mixing["append"], msg
            mob_values[loc_key] = loc_mixing

        # Interpret user requests if specified in a way other than absolute mobility values
        if loc_mixing:
            loc_mixing["values"] = parse_values(loc_mixing["values"])

    # Reformat data so that we only have times and values as the keys within each location key, without append
    return {k: pd.Series(v["values"], index=v["times"]) for k, v in mob_values.items()}


def get_mobility_specific_period(
    country: str,
    region: str,
    google_mobility_locations: Dict[str, Dict[str, float]],
    split_dates: List[float],
) -> Tuple[pd.DataFrame, dict]:
    """
    Loads google mobility data, splits it for the requested time duration and then returns a mobility function for each
    location.

    Args:
        country: Country request
        region: Any sub-region of the country
        google_mobility_locations: The mapping instructions from modelled locations (first keys) to Google locations (second keys)
        split_dates: The dates at which we want to split the data

    Returns:
        The truncated mobility dates and values

    """

    google_mob_data, google_mobility_days = get_mobility_data(country, region, COVID_BASE_DATETIME)
    mobility_values = weight_mobility_data(google_mob_data, google_mobility_locations)

    start_idx = google_mobility_days.index(split_dates[0])
    end_idx = google_mobility_days.index(split_dates[1])

    mob_dates = google_mobility_days[start_idx:end_idx]
    mob_values = {
        loc: mobility_values.loc[start_idx : end_idx - 1, loc].to_list()
        for loc in mobility_values.columns
    }
    return mob_dates, mob_values


def parse_values(values: List[float]) -> List[float]:
    """
    Convert all mixing time series values to a float, using the functions below.

    Args:
        values: The starting values, which may include function-based user requests

    Returns:
        The processed values, now all floats regardless of how they started out

    """
    new_values = []
    for v in values:

        # Apply a function to the history to get the next value
        if type(v) is list:
            func_name = v[0]
            args = [new_values] + v[1:]
            func = PARSE_FUNCS[func_name]
            new_val = func(*args)

        # Otherwise leave unchanged
        else:
            new_val = v

        assert new_val >= 0.0, f"Mobility value less than zero: {new_val}"
        new_values.append(new_val)

    return new_values


def repeat_prev(prev_vals: List[float]) -> float:
    """
    Repeats the last previously seen value again.

    """

    return prev_vals[-1]


def add_to_prev(prev_vals: List[float], increment: float) -> float:
    """
    Add increment to previous.

    """

    return prev_vals[-1] + increment


def add_to_prev_up_to_1(prev_vals: List[float], increment: float) -> float:
    """
    Add increment to previous, but set ceiling at 1.

    """

    val = prev_vals[-1] + increment
    if val > 1.0:
        return 1.0
    else:
        return val


def scale_prev(prev_vals: List[float], fraction: float) -> float:
    """
    Apply a multiplier to the previous value.

    """

    return prev_vals[-1] * fraction


def scale_prev_up_to_1(prev_vals: List[float], multiplier: float) -> float:
    """
    Apply a multiplier to the previous value, saturating at one.

    """

    val = prev_vals[-1] * multiplier
    return val if val < 1.0 else 1.0


def close_gap_to_1(prev_vals: List[float], fraction: float) -> float:
    """
    Reduce the difference between the last value and one (full mobility) according to the "fraction" request.

    """

    prev_val = prev_vals[-1]
    return (1.0 - prev_val) * fraction + prev_val


def max_last_period(prev_vals: List[float], period: int) -> float:
    """
    The maximum mobility estimate observed over the preceding days.

    """

    last_n_values = min(len(prev_vals), period)
    return max(prev_vals[-last_n_values:])


def min_last_period(prev_vals: List[float], period: int) -> float:
    """
    The minimum mobility estimate observed over the preceding days.

    """

    last_n_values = min(len(prev_vals), period)
    return min(prev_vals[-last_n_values:])


def average_mobility(prev_vals: List[float], period: int) -> float:
    """
    The average mobility estimate over the preceding days.

    """

    last_n_values = min(len(prev_vals), period)
    return sum(prev_vals[-last_n_values:]) / len(prev_vals[-last_n_values:])


def copy_mobility(prev_vals: List[float], ignore_vals: int) -> float:
    """
    Returns the mobility level at the requested time by ignoring the last values defined by ignore_vals.

    """

    prev_val = prev_vals[-ignore_vals + 1 :]
    return prev_val[1]


def close_to_max_last_period(prev_vals: List[float], period: int, fraction: float) -> float:
    """
    Partial return from last mobility estimate to the highest level observed over the recent period specified.

    """

    last_n_values = min(len(prev_vals), period)
    max_val_last_period = max(prev_vals[-last_n_values:])
    prev_val = prev_vals[-1]
    return (max_val_last_period - prev_val) * fraction + prev_val


# Specific mobility estimates used for the Philippines
CQ_MOBILITY = {
    # GCQ Reference period: from May 15 2021
    "GCQ": {"work": 0.70, "other_locations": 0.70},
    # MECQ Reference period April 12-May 15 2021
    "MECQ": {"work": 0.55, "other_locations": 0.55},
    # ECQ Reference period March 29-April 11 2021
    "ECQ": {"work": 0.40, "other_locations": 0.40},
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
    "average_mobility": average_mobility,
    # Used for the Philippines
    "ECQ": ecq,
    "MECQ": mecq,
    "GCQ": gcq,
}
