from autumn.constants import Region
from autumn.tool_kit.model_register import AppRegion

from apps.covid_19.model import build_model

from .calibrate import run_calibration_chain

sabah_region = AppRegion(
    app_name="covid_19",
    region_name=Region.SABAH,
    build_model=build_model,
    calibrate_model=run_calibration_chain,
)
