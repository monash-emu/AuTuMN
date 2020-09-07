from autumn.constants import Region
from autumn.tool_kit.params import load_targets


def get_all_vic_notifications(excluded_regions=()):
    excluded_regions = [r.lower().replace("_", "-") for r in excluded_regions]
    import_aggs = None
    for region in [r for r in Region.VICTORIA_SUBREGIONS if r not in excluded_regions and r != "victoria"]:
        import_times, import_values = \
            get_region_notifications(region)
        import_aggs = \
            [i + j for i, j in zip(import_aggs, import_values)] if import_aggs else import_values
    return import_times, import_aggs


def get_region_notifications(region):
    notifications = load_targets("covid_19", region.lower().replace("-", "_"))["notifications"]
    return notifications["times"], notifications["values"]
