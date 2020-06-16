import os
import numpy as np
import pandas as pd
from functools import lru_cache

from autumn.constants import DATA_PATH

DATA_FOLDER = os.path.join(DATA_PATH, "social_mixing")
LOCATIONS = ("all_locations", "home", "other_locations", "school", "work")

# Map regions to countries
COUNTRY_MAPPING = {
    "victoria": "australia",
    "manila": "philippines",
    "calabarzon": "philippines",
    "bicol": "philippines",
    "central-visayas": "philippines",
}

# Cache result beecause this gets called 1000s of times during calibration.
@lru_cache(maxsize=None)
def load_country_mixing_matrix(mixing_location: str, country: str):
    """
    Load a mixing matrix sheet, according to name of the sheet (i.e. country)
    See the README in data/social_mixing for more info.
    """
    assert mixing_location in LOCATIONS, f"Invalid mixing location {mixing_location}"
    country = COUNTRY_MAPPING.get(country, country)
    if country.title() < "Mozambique":
        # Files with name ending with _1 have a header
        sheet_number = "1"
        header_argument = 0
    else:
        # but not those ending with _2 - plus need to determine file to read
        sheet_number = "2"
        header_argument = None

    sheet_name = f"MUestimates_{mixing_location}_{sheet_number}.xlsx"
    file_dir = os.path.join(DATA_FOLDER, sheet_name)
    df = pd.read_excel(file_dir, sheet_name=country.title(), header=header_argument)
    return np.array(df)
