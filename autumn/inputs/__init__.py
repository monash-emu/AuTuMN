from autumn.tool_kit import Timer

from .database import build_input_database
from .mobility.fetch import fetch_mobility_data
from .demography.queries import (
    get_crude_birth_rate,
    get_population_by_agegroup,
    get_iso3_from_country_name,
    get_death_rates_by_agegroup,
    get_crude_birth_rate,
)
from .social_mixing.queries import get_country_mixing_matrix
from .mobility.queries import get_mobility_data


def fetch_input_data():
    """
    Fetch input data from external sources,
    which can then be used to build the input database.
    """
    with Timer("Fetching mobility data."):
        fetch_mobility_data()
