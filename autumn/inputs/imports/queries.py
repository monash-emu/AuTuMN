import json
import os

from autumn.settings import Region
from autumn.settings import INPUT_DATA_PATH

IMPORTS_FILE = os.path.join(INPUT_DATA_PATH, "imports", "imports.secret.json")


def get_region_imports(region_name: str):
    with open(IMPORTS_FILE, "r") as f:
        imports_data = json.load(f)

    region_data = imports_data[region_name]
    times, values = region_data["times"], region_data["values"]
    return times, values


def get_all_vic_region_imports():
    import_aggregates = None
    for region in [r for r in Region.VICTORIA_SUBREGIONS if r != "victoria"]:
        import_times, import_values = get_region_imports(region)
        import_aggregates = (
            [i + j for i, j in zip(import_values, import_aggregates)]
            if import_aggregates
            else import_values
        )
    return import_times, import_aggregates
