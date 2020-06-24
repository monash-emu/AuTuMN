"""
Constants used in building the AuTuMN / SUMMER models.
"""
import os

# Import summer constants here for convenience
from summer.constants import (
    BirthApproach,
    IntegrationType,
    Stratification,
    Compartment,
    Flow,
)

# Filesystem paths
file_path = os.path.abspath(__file__)
separator = "\\" if "\\" in file_path else "/"
BASE_PATH = separator.join(file_path.split(separator)[:-2])
DATA_PATH = os.path.join(BASE_PATH, "data")
APPS_PATH = os.path.join(BASE_PATH, "apps")
EXCEL_PATH = os.path.join(DATA_PATH, "xls")


class Region:
    AUSTRALIA = "australia"
    PHILIPPINES = "philippines"
    MALAYSIA = "malaysia"
    VICTORIA = "victoria"
    NSW = "nsw"
    LIBERIA = "liberia"
    MANILA = "manila"
    CALABARZON = "calabarzon"
    BICOL = "bicol"
    CENTRAL_VISAYAS = "central-visayas"
    UNITED_KINGDOM = "united-kingdom"
    REGIONS = [
        AUSTRALIA,
        PHILIPPINES,
        MALAYSIA,
        VICTORIA,
        NSW,
        LIBERIA,
        MANILA,
        CALABARZON,
        BICOL,
        CENTRAL_VISAYAS,
        UNITED_KINGDOM,
    ]
    REGION_COUNTRY = {
        UNITED_KINGDOM: UNITED_KINGDOM,
        MALAYSIA: MALAYSIA,
        LIBERIA: LIBERIA,
        AUSTRALIA: AUSTRALIA,
        VICTORIA: AUSTRALIA,
        NSW: AUSTRALIA,
        PHILIPPINES: PHILIPPINES,
        MANILA: PHILIPPINES,
        CALABARZON: PHILIPPINES,
        BICOL: PHILIPPINES,
        CENTRAL_VISAYAS: PHILIPPINES,
    }

    @staticmethod
    def get_country(region: str):
        return REGION_COUNTRY[region]
