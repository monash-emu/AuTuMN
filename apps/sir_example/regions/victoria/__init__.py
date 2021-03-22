from apps.sir_example.model import build_model
from autumn.region import Region
from autumn.utils.model_register import AppRegion

from .calibrate import run_calibration_chain

victoria_region = AppRegion(
    app_name="sir_example",
    region_name=Region.VICTORIA,
    build_model=build_model,
    calibrate_model=run_calibration_chain,
)
