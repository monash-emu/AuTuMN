import os
import json

from autumn.constants import BASE_PATH, Region


def get_region_imports(region_name):
    imports_path = os.path.join(BASE_PATH, "data\\inputs\\imports")
    region_filename = region_name.replace("-", "_")
    imports_filename = os.path.join(imports_path, f"{region_filename}.secret.json")
    with open(imports_filename, "r") as file:
        import_notifications = json.load(file)["notifications"]
        times, values = import_notifications["times"], import_notifications["values"]
    return times, values


def get_all_vic_region_imports():
    import_aggregates = None
    for region in Region.VICTORIA_SUBREGIONS:
        import_times, import_values = \
            get_region_imports(region)
        import_aggregates = \
            [i + j for i, j in zip(import_values, import_aggregates)] if import_aggregates else import_values
    return import_times, import_aggregates
