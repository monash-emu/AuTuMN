import os
import numpy as np
import pandas as pd
from functools import lru_cache

from autumn.inputs.database import get_input_db


LOCATIONS = ("all_locations", "home", "other_locations", "school", "work")


# Cache result beecause this gets called 1000s of times during calibration.
@lru_cache(maxsize=None)
def get_country_mixing_matrix(mixing_location: str, country_iso_code: str):
    """
    Load a mixing matrix for a given country and mixing location.
    The rows and columns indices of each matrix represent a 5 year age bracket from 0-80,
    giving us a 16x16 matrix.
    """
    assert mixing_location in LOCATIONS, f"Invalid mixing location {mixing_location}"
    input_db = get_input_db()
    cols = [f"X{n}" for n in range(1, 17)]
    mix_df = input_db.query(
        "social_mixing",
        column=",".join(cols),
        conditions=[f"iso3='{country_iso_code}'", f"location='{mixing_location}'",],
    )
    matrix = np.array(mix_df)
    assert matrix.shape == (16, 16), "Mixing matrix is not 16x16"
    return matrix
