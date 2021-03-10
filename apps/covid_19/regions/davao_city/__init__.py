from autumn.region import Region
from autumn.utils.model_register import AppRegion

from apps.covid_19.model import build_model

from .calibrate import run_calibration_chain

davao_city_region = AppRegion(
    app_name="covid_19",
    region_name=Region.DAVAO_CITY,
    build_model=build_model,
    calibrate_model=run_calibration_chain,
)
