from typing import List
from autumn.curve import scale_up_function
from autumn.constants import Region
from autumn.tool_kit.params import load_targets


def get_importation_rate_func_as_birth_rates(
    importation_times: List[float],
    importation_n_cases: List[float],
    detect_prop_func,
):
    """
    When imported cases are explicitly simulated as part of the modelled population. They enter the late_infectious
    compartment through a birth process.
    """

    # Inflate importation numbers to account for undetected cases (assumed to be asymptomatic or sympt non hospital)
    actual_imported_cases = [
        importation_n_cases[i_time] / detect_prop_func(time)
        for i_time, time in enumerate(importation_times)
    ]

    # Scale-up curve for importation numbers
    recruitment_rate = scale_up_function(
        importation_times, actual_imported_cases, method=4, smoothness=5.0, bound_low=0.0
    )
    return recruitment_rate


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
        import_times, import_values = get_region_notifications(region)
        import_aggs = (
            [i + j for i, j in zip(import_aggs, import_values)] if import_aggs else import_values
        )
    return import_times, import_aggs


def get_region_notifications(region):
    """
    Grab the notifications, hijacking the code and files that store the target values for use in calibration
    """

    notifications = load_targets("covid_19", region.lower().replace("-", "_"))["notifications"]
    return notifications["times"], notifications["values"]
