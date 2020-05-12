import itertools
from datetime import date
from summer.model.utils.string import find_name_components

from autumn.constants import Compartment


def find_incidence_outputs(parameters):
    last_presympt = (
        Compartment.PRESYMPTOMATIC + "_" + str(parameters["n_compartment_repeats"][Compartment.PRESYMPTOMATIC])
        if parameters["n_compartment_repeats"][Compartment.PRESYMPTOMATIC] > 1
        else Compartment.PRESYMPTOMATIC
    )
    first_infectious = (
        Compartment.EARLY_INFECTIOUS + "_1" if
        parameters["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS] > 1 else
        Compartment.EARLY_INFECTIOUS
    )
    return {
        "incidence": {
            "origin": last_presympt,
            "to": first_infectious,
            "origin_condition": "",
            "to_condition": "",
        }
    }


def create_fully_stratified_incidence_covid(requested_stratifications, strata_dict, model_params):
    """
    Create derived outputs for fully disaggregated incidence
    """
    out_connections = {}
    origin_compartment = (
        Compartment.PRESYMPTOMATIC
        if model_params["n_compartment_repeats"][Compartment.PRESYMPTOMATIC] < 2
        else Compartment.PRESYMPTOMATIC
        + "_"
        + str(model_params["n_compartment_repeats"][Compartment.PRESYMPTOMATIC])
    )
    to_compartment = (
        Compartment.EARLY_INFECTIOUS
        if model_params["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS] < 2
        else Compartment.EARLY_INFECTIOUS + "_1"
    )

    all_tags_by_stratification = []
    for stratification in requested_stratifications:
        this_stratification_tags = []
        for stratum in strata_dict[stratification]:
            this_stratification_tags.append(stratification + "_" + stratum)
        all_tags_by_stratification.append(this_stratification_tags)

    all_tag_lists = list(itertools.product(*all_tags_by_stratification))

    for tag_list in all_tag_lists:
        stratum_name = "X".join(tag_list)
        out_connections["incidenceX" + stratum_name] = {
            "origin": origin_compartment,
            "to": to_compartment,
            "origin_condition": "",
            "to_condition": stratum_name,
        }

    return out_connections


def create_fully_stratified_progress_covid(requested_stratifications, strata_dict, model_params):
    """
    Create derived outputs for fully disaggregated incidence
    """
    out_connections = {}
    origin_compartment = (
        Compartment.EARLY_INFECTIOUS
        if model_params["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS] < 2
        else Compartment.EARLY_INFECTIOUS
        + "_"
        + str(model_params["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS])
    )
    to_compartment = (
        Compartment.LATE_INFECTIOUS
        if model_params["n_compartment_repeats"][Compartment.LATE_INFECTIOUS] < 2
        else Compartment.LATE_INFECTIOUS + "_1"
    )

    all_tags_by_stratification = []
    for stratification in requested_stratifications:
        this_stratification_tags = []
        for stratum in strata_dict[stratification]:
            this_stratification_tags.append(stratification + "_" + stratum)
        all_tags_by_stratification.append(this_stratification_tags)

    all_tag_lists = list(itertools.product(*all_tags_by_stratification))

    for tag_list in all_tag_lists:
        stratum_name = "X".join(tag_list)
        out_connections["progressX" + stratum_name] = {
            "origin": origin_compartment,
            "to": to_compartment,
            "origin_condition": "",
            "to_condition": stratum_name,
        }

    return out_connections


def calculate_notifications_covid(model, time):
    """
    Returns the number of notifications for a given time.
    The fully stratified incidence outputs must be available before calling this function
    """
    notifications = 0.0
    this_time_index = model.times.index(time)
    for key, value in model.derived_outputs.items():
        if "progressX" in key and \
                any([stratum in key for stratum in model.all_stratifications["clinical"][2:]]):
            notifications += value[this_time_index]

    if model.parameters['implement_importation'] and model.parameters['imported_cases_explict']:
        prop_imported_detected = model.parameters['symptomatic_props_imported'] * \
                        (model.parameters['prop_isolated_among_symptomatic_imported'] +
                         model.parameters['hospital_props_imported'])
        notifications += model.time_variants["crude_birth_rate"](time) *\
                          sum(model.compartment_values) * prop_imported_detected

    return notifications


def calculate_incidence_icu_covid(model, time):
    this_time_index = model.times.index(time)
    incidence_icu = 0.
    for key, value in model.derived_outputs.items():
        if "incidence" in find_name_components(key) and "clinical_icu" in find_name_components(key):
            incidence_icu += value[this_time_index]
    return incidence_icu


def find_date_from_year_start(times, incidence):
    """
    Messy patch to shift dates over such that zero represents the start of the year and the number of cases are
    approximately correct for Australia at 22nd March
    """
    year, month, day = 2020, 3, 22
    cases = 1098.0
    data_days_from_year_start = (date(year, month, day) - date(year, 1, 1)).days
    model_days_reach_target = next(i_inc[0] for i_inc in enumerate(incidence) if i_inc[1] > cases)
    print(f"Integer date at which target reached is: {model_days_reach_target}")
    days_to_add = data_days_from_year_start - model_days_reach_target
    return [int(i_time) + days_to_add for i_time in times]
