from autumn.constants import Region
from autumn.tool_kit.params import load_targets


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

    # Loop over all the health system cluster sub-regions, ignoring the excluded regions and the state as a whole
    for region in [r for r in Region.VICTORIA_SUBREGIONS if r not in excluded_regions and r != "victoria"]:
        import_times, import_values = \
            get_region_notifications(region)
        import_aggs = \
            [i + j for i, j in zip(import_aggs, import_values)] if import_aggs else import_values
    return import_times, import_aggs


def get_region_notifications(region):
    """
    Grab the notifications, hijacking the code and files that store the target values for use in calibration
    """

    notifications = load_targets("covid_19", region.lower().replace("-", "_"))["notifications"]
    return notifications["times"], notifications["values"]
