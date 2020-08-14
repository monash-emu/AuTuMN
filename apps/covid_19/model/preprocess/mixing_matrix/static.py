import numpy as np

from autumn.inputs import get_country_mixing_matrix


def build_static(country_iso3: str) -> np.ndarray:
    """
    Get a non-time-varying mixing matrix for a country.
    Each row/column represents a 5 year age bucket.
    Matrix is 16x16, representing ages 0-80+
    """
    return get_country_mixing_matrix("all_locations", country_iso3)
