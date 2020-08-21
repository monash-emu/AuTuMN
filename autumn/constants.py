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
INPUT_DATA_PATH = os.path.join(DATA_PATH, "inputs")
OUTPUT_DATA_PATH = os.path.join(DATA_PATH, "outputs")
APPS_PATH = os.path.join(BASE_PATH, "apps")


class Region:
    AUSTRALIA = "australia"
    PHILIPPINES = "philippines"
    MALAYSIA = "malaysia"
    VICTORIA = "victoria"
    WEST_MELBOURNE = "west_melbourne"
    NORTH_MELBOURNE = "north_melbourne"
    NSW = "nsw"
    LIBERIA = "liberia"
    MANILA = "manila"
    CALABARZON = "calabarzon"
    BICOL = "bicol"
    CENTRAL_VISAYAS = "central-visayas"
    UNITED_KINGDOM = "united-kingdom"
    BELGIUM = "belgium"
    ITALY = "italy"
    SWEDEN = "sweden"
    FRANCE = "france"
    SPAIN = "spain"
    NORTH_METRO ="north_metro"
    SOUTH_EAST_METRO = "south_east_metro"
    SOUTH_METRO = "south_metro"
    WEST_METRO = "west_metro"
    BARWON_SOUTH_WEST = "barwon_south_west"
    GIPPSLAND = "gippsland"
    HUME = "hume"
    LODDON_MALLEE = "loddon_mallee"
    GRAMPIANS = "grampians"

    REGIONS = [
        AUSTRALIA,
        PHILIPPINES,
        MALAYSIA,
        VICTORIA,
        WEST_MELBOURNE,
        NORTH_MELBOURNE,
        NSW,
        LIBERIA,
        MANILA,
        CALABARZON,
        BICOL,
        CENTRAL_VISAYAS,
        UNITED_KINGDOM,
        BELGIUM,
        ITALY,
        SWEDEN,
        FRANCE,
        SPAIN,
        NORTH_METRO,
        SOUTH_EAST_METRO,
        SOUTH_METRO,
        WEST_METRO,
        BARWON_SOUTH_WEST,
        GIPPSLAND,
        HUME,
        LODDON_MALLEE,
        GRAMPIANS,
    ]
