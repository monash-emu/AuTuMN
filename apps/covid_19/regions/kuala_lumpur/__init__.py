from apps.covid_19.model import build_model
from autumn.region import Region
from autumn.utils.model_register import AppRegion

from .calibrate import run_calibration_chain

kuala_lumpur_region = AppRegion(
    app_name="covid_19",
    region_name=Region.KUALA_LUMPUR,
    build_model=build_model,
    calibrate_model=run_calibration_chain,
)
