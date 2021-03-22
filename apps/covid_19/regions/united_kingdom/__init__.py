from apps.covid_19.model import build_model
from autumn.region import Region
from autumn.utils.model_register import AppRegion

from .calibrate import run_calibration_chain

united_kingdom_region = AppRegion(
    app_name="covid_19",
    region_name=Region.UNITED_KINGDOM,
    build_model=build_model,
    calibrate_model=run_calibration_chain,
)
