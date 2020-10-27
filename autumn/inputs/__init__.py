from autumn.tool_kit import Timer

from .database import build_input_database
from .mobility.fetch import fetch_mobility_data
from .covid_au.fetch import fetch_covid_au_data
from .covid_phl.fetch import fetch_covid_phl_data
from .john_hopkins.fetch import fetch_john_hopkins_data

from .demography.queries import (
    get_crude_birth_rate,
    get_population_by_agegroup,
    get_iso3_from_country_name,
    get_death_rates_by_agegroup,
    get_life_expectancy_by_agegroup,
    get_crude_birth_rate,
)
from .social_mixing.queries import get_country_mixing_matrix
from .mobility.queries import get_mobility_data
from .john_hopkins.queries import get_john_hopkins_data
from .covid_au.queries import get_vic_testing_numbers
from .covid_phl.queries import get_phl_testing_numbers


def fetch_input_data():
    """
    Fetch input data from external sources,
    which can then be used to build the input database.
    """
    with Timer("Fetching mobility data."):
        fetch_mobility_data()

    with Timer("Fetching John Hopkins data."):
        fetch_john_hopkins_data()

    with Timer("Fetching COVID AU data."):
        fetch_covid_au_data()

    with Timer("Fetching COVID PHL data."):
        fetch_covid_phl_data()
