from autumn.tools.utils.utils import update_mle_from_remote_calibration
from autumn.settings.region import Region


# Specify run ids. If None, the latest run will be considered.
run_ids = {
    "calabarzon": None,
    "davao-region": None,
    "central-visayas": None,
    "manila": None,
    "philippines": None,
}


if __name__ == "__main__":
    for region in Region.PHILIPPINES_REGIONS:
        update_mle_from_remote_calibration("covid_19", region, run_ids[region])
