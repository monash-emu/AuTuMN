from autumn.constants import Region
from . import (
    australia,
    philippines,
)

CALIBRATIONS = {
    Region.AUSTRALIA: australia.run_calibration_chain,
    Region.PHILIPPINES: philippines.run_calibration_chain,
}


def get_calibration_func(region: str):
    return CALIBRATIONS[region]
