from typing import Callable, Dict, List

from autumn.models.covid_19.constants import BASE_DATETIME
from autumn.models.covid_19.parameters import Country, MixingLocation
from autumn.tools.curve import scale_up_function
from autumn.tools.inputs.mobility.queries import get_mobility_data
from autumn.tools.utils.utils import apply_moving_average

LOCATIONS = ["home", "other_locations", "school", "work"]


def weight_mobility_data(mob_df, location_map):
    """
    Get weighted average for each modelled location from Google mobility estimates.
    """

    revised_location_map = {key: value for key, value in location_map.items() if value}
    for model_loc, google_locs in revised_location_map.items():
        mob_df[model_loc] = 0
        for g_loc in google_locs:
            mob_df[model_loc] += mob_df[g_loc] * revised_location_map[model_loc][g_loc]

    return mob_df[revised_location_map.keys()]


def get_mobility_funcs(
    country: Country,
    region: str,
    mixing: Dict[str, MixingLocation],
    google_mobility_locations: List[str],
    npi_effectiveness_params: Dict[str, float],
    square_mobility_effect: bool,
    smooth_google_data: bool,
) -> Dict[str, Callable[[float], float]]:
    """
    Loads google mobility data, combines it with user requested timeseries data
    and then returns a mobility function for each location.
    """

    mob_df, google_mobility_days = get_mobility_data(country.iso3, region, BASE_DATETIME)
    mob_values_df = weight_mobility_data(mob_df, google_mobility_locations)

    if smooth_google_data:
        for loc in mob_values_df.columns:
            mob_values_df[loc] = apply_moving_average(mob_values_df[loc], 7)

    google_mobility_values = mob_values_df.to_dict(orient="list")

    # Build mixing data timeseries
    mixing = update_mixing_data(
        {k: v.dict() for k, v in mixing.items()},
        npi_effectiveness_params,
        google_mobility_values,
        google_mobility_days,
    )

    # Build the time variant location-specific macrodistancing adjustment functions from mixing timeseries
    mobility_funcs = {}
    for location, timeseries in mixing.items():
        if square_mobility_effect:
            loc_vals = [v ** 2 for v in timeseries["values"]]
        else:
            loc_vals = timeseries["values"]
        mobility_funcs[location] = scale_up_function(timeseries["times"], loc_vals, method=4)

    return mobility_funcs


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

    mob_df, google_mobility_days = get_mobility_data(country, region, BASE_DATETIME)
    mob_values_df = weight_mobility_data(mob_df, google_mobility_locations)

    first_timepoint_index = google_mobility_days.index(split_dates[0])
    second_timepoint_index = google_mobility_days.index(split_dates[1])

    split_google_mobility_days = google_mobility_days[first_timepoint_index: second_timepoint_index]
    spilt_google_mobility_values = {
        loc: mob_values_df.loc[first_timepoint_index: second_timepoint_index - 1, loc].to_list() for
        loc in mob_values_df.columns
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
