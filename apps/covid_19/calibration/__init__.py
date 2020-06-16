from autumn.constants import Region
from . import (
    bicol,
    calabarzon,
    central_visayas,
    liberia,
    malaysia,
    manila,
    philippines,
    victoria,
)

CALIBRATIONS = {
    Region.BICOL: bicol.run_calibration_chain,
    Region.CALABARZON: calabarzon.run_calibration_chain,
    Region.CENTRAL_VISAYAS: central_visayas.run_calibration_chain,
    Region.LIBERIA: liberia.run_calibration_chain,
    Region.MALAYSIA: malaysia.run_calibration_chain,
    Region.MANILA: manila.run_calibration_chain,
    Region.PHILIPPINES: philippines.run_calibration_chain,
    Region.VICTORIA: victoria.run_calibration_chain,
}


def get_calibration_func(region: str):
    return CALIBRATIONS[region]
