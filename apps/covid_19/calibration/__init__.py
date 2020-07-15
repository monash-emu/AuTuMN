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
    nsw,
    united_kingdom,
    belgium,
    italy,
    sweden,
    france,
    spain,
)

CALIBRATIONS = {
    Region.BICOL: bicol.run_calibration_chain,
    Region.CALABARZON: calabarzon.run_calibration_chain,
    Region.CENTRAL_VISAYAS: central_visayas.run_calibration_chain,
    # No Google Mobility data for Liberia
    # Region.LIBERIA: liberia.run_calibration_chain,
    Region.MALAYSIA: malaysia.run_calibration_chain,
    Region.MANILA: manila.run_calibration_chain,
    Region.PHILIPPINES: philippines.run_calibration_chain,
    Region.VICTORIA: victoria.run_calibration_chain,
    Region.NSW: nsw.run_calibration_chain,
    Region.UNITED_KINGDOM: united_kingdom.run_calibration_chain,
    Region.BELGIUM: belgium.run_calibration_chain,
    Region.ITALY: italy.run_calibration_chain,
    Region.SWEDEN: sweden.run_calibration_chain,
    Region.FRANCE: france.run_calibration_chain,


}


def get_calibration_func(region: str):
    return CALIBRATIONS[region]
