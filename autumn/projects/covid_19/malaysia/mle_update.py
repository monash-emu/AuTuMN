from autumn.tools.utils.utils import update_mle_from_remote_calibration
from autumn.settings.region import Region


# Specify chosen run ids. If None, the latest run found on the server will be considered.
run_ids = {
    "malaysia": None,
    "selangor": None,
    "johor": None,
    "penang": None,
    "kuala-lumpur": None,
}


if __name__ == "__main__":
    for region in Region.MALAYSIA_REGIONS:
        update_mle_from_remote_calibration("covid_19", region, run_ids[region])
