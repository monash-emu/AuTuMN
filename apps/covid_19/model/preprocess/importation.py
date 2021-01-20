from typing import List
from autumn.curve import scale_up_function
from autumn.constants import Region
from autumn.tool_kit.params import load_targets

from apps.covid_19.model.parameters import Parameters, Importation
from apps.covid_19.model.preprocess.clinical import get_proportion_symptomatic
from apps.covid_19.model.preprocess.case_detection import (
    build_detected_proportion_func,
    get_testing_pop,
)


def build_importation_rate_func(
    params: Parameters, agegroup_strata: List[str], total_pops: List[int]
):
    pop = params.population
    country = params.country

    is_region_vic = pop.region and Region.to_name(pop.region) in Region.VICTORIA_SUBREGIONS
    if is_region_vic:
        import_times, importation_data = get_all_vic_notifications(excluded_regions=(pop.region,))
        testing_pop, _ = get_testing_pop(agegroup_strata, country, pop)
        movement_to_region = sum(total_pops) / sum(testing_pop) * params.importation.movement_prop
        import_cases = [i_cases * movement_to_region for i_cases in importation_data]
    else:
        import_times = params.importation.case_timeseries.times
        import_cases = params.importation.case_timeseries.values

    get_abs_detection_proportion_imported = build_abs_detection_proportion_imported(
        params, agegroup_strata
    )

    # Inflate importation numbers to account for undetected cases (assumed to be asymptomatic or sympt non hospital)
    actual_imported_cases = [
        import_cases[i_time] / get_abs_detection_proportion_imported(time)
        for i_time, time in enumerate(import_times)
    ]

    # Scale-up curve for importation numbers
    recruitment_rate = scale_up_function(
        import_times, actual_imported_cases, method=4, smoothness=5.0, bound_low=0.0
    )
    return recruitment_rate


def build_abs_detection_proportion_imported(params: Parameters, agegroup_strata: List[str]):
    """
    Returns a function which returns absolute proprotion of imported people who are detected to be infectious.
    """
    pop = params.population
    country = params.country

    # Get case detection rate function.
    get_detected_proportion = build_detected_proportion_func(
        agegroup_strata, country, pop, params.testing_to_detection, params.case_detection
    )
    # Determine how many importations there are, including the undetected and asymptomatic importations
    symptomatic_props = get_proportion_symptomatic(params)
    importation_props_by_age = get_importation_props_by_age(params.importation, agegroup_strata)
    import_symptomatic_prop = sum(
        [
            import_prop * sympt_prop
            for import_prop, sympt_prop in zip(importation_props_by_age.values(), symptomatic_props)
        ]
    )

    def get_abs_detection_proportion_imported(t):
        # Returns absolute proprotion of imported people who are detected to be infectious.
        return import_symptomatic_prop * get_detected_proportion(t)

    return get_abs_detection_proportion_imported


def get_importation_props_by_age(importation: Importation, agegroup_strata: List[str]):
    """
    Get proportion of importations by age. This is used to calculate case detection and used in age stratification.
    """
    if importation and importation.props_by_age:
        return importation.props_by_age
    else:
        return {s: 1.0 / len(agegroup_strata) for s in agegroup_strata}


def get_all_vic_notifications(excluded_regions=()):
    """
    Get all Victorian notifications for use in seeding the epidemic as an "importation" function, to represent movement
    between other regions and the index health cluster region being modelled.

    :param excluded_regions: tuple
        Any regions that should be excluded, here expected to be the index cluster being modelled
    :return:
    import_times: list
        Times corresponding to the series of notifications
    import_aggs: list
        The series of notification values
    """

    # FIXME: This should be generalised to a function somewhere to convert between the two different string formats used
    excluded_regions = [r.lower().replace("_", "-") for r in excluded_regions]
    import_aggs = None
    import_times = None

    # Loop over all the health system cluster sub-regions, ignoring the excluded regions and the state as a whole
    for region in [
        r for r in Region.VICTORIA_SUBREGIONS if r not in excluded_regions and r != "victoria"
    ]:
        import_times, import_values = _get_region_notifications(region)
        import_aggs = (
            [i + j for i, j in zip(import_aggs, import_values)] if import_aggs else import_values
        )
    return import_times, import_aggs


def _get_region_notifications(region):
    """
    Grab the notifications, hijacking the code and files that store the target values for use in calibration
    """

    notifications = load_targets("covid_19", region.lower().replace("-", "_"))["notifications"]
    return notifications["times"], notifications["values"]
