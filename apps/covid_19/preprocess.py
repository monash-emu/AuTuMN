"""
Pre-processing of model parameters
"""
from autumn.constants import Compartment
from autumn.tool_kit.utils import find_relative_date_from_string_or_tuple
from autumn.demography.ageing import add_agegroup_breaks


def preprocess_params(params: dict, update_params: dict):

    # Revise any dates for mixing matrices submitted in YMD format
    if params.get("mixing"):
        params["mixing"] = revise_mixing_data_for_dicts(params["mixing"])
        revise_dates_if_ymd(params["mixing"])

    params = add_agegroup_breaks(params)

    # Update, not used in single application run
    params.update(update_params)

    # Update parameters stored in dictionaries that need to be modified during calibration
    params = update_dict_params_for_calibration(params)

    return params


def get_mixing_lists_from_dict(working_dict):
    return [i_key for i_key in working_dict.keys()], [i_key for i_key in working_dict.values()]


def revise_mixing_data_for_dicts(parameters):
    list_of_possible_keys = ["home", "other_locations", "school", "work"]
    for age_index in range(16):
        list_of_possible_keys.append("age_" + str(age_index))
    for mixing_key in list_of_possible_keys:
        if mixing_key in parameters:
            (
                parameters[mixing_key + "_times"],
                parameters[mixing_key + "_values"],
            ) = get_mixing_lists_from_dict(parameters[mixing_key])
    return parameters


def revise_dates_if_ymd(mixing_params):
    """
    Find any mixing times parameters that were submitted as a three element list of year, month day - and revise to an
    integer representing the number of days from the reference time.
    """
    for key in (k for k in mixing_params if k.endswith("_times")):
        for i_time, time in enumerate(mixing_params[key]):
            if isinstance(time, (list, str)):
                mixing_params[key][i_time] = find_relative_date_from_string_or_tuple(time)


def update_dict_params_for_calibration(params):
    """
    Update some specific parameters that are stored in a dictionary but are updated during calibration.
    For example, we may want to update params['default']['compartment_periods']['incubation'] using the parameter
    ['default']['compartment_periods_incubation']
    :param params: dict
        contains the model parameters
    :return: the updated dictionary
    """

    if "n_imported_cases_final" in params:
        params["data"]["n_imported_cases"][-1] = params["n_imported_cases_final"]

    for location in ["school", "work", "home", "other_locations"]:
        if "npi_effectiveness_" + location in params:
            params["npi_effectiveness"][location] = params["npi_effectiveness_" + location]

    for comp_type in [
        "incubation",
        "infectious",
        "late",
        "hospital_early",
        "hospital_late",
        "icu_early",
        "icu_late",
    ]:
        if "compartment_periods_" + comp_type in params:
            params["compartment_periods"][comp_type] = params["compartment_periods_" + comp_type]

    return params
