from apps.tuberculosis_strains.model import build_model
from autumn.region import Region
from autumn.utils.model_register import AppRegion

from .calibrate import run_calibration_chain

philippines_region = AppRegion(
    app_name="tuberculosis_strains",
    region_name=Region.PHILIPPINES,
    build_model=build_model,
    calibrate_model=run_calibration_chain,
)
