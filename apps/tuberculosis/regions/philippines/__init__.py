from autumn.region import Region
from autumn.tool_kit.model_register import AppRegion

from apps.tuberculosis.model import build_model

from .calibrate import run_calibration_chain

philippines_region = AppRegion(
    app_name="tuberculosis",
    region_name=Region.PHILIPPINES,
    build_model=build_model,
    calibrate_model=run_calibration_chain,
)
