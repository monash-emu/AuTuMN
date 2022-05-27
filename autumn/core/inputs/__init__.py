from autumn.core.inputs.covid_au.fetch import fetch_covid_au_data
from autumn.core.inputs.covid_au.queries import get_vic_testing_numbers
from autumn.core.inputs.covid_phl.fetch import fetch_covid_phl_data
from autumn.core.inputs.covid_mys.fetch import fetch_covid_mys_data
from autumn.core.inputs.covid_mmr.fetch import fetch_covid_mmr_data

from autumn.core.inputs.covid_survey.fetch import fetch_covid_survey_data
from autumn.core.inputs.covid_phl.queries import get_phl_subregion_testing_numbers
from autumn.core.inputs.database import build_input_database
from autumn.core.inputs.demography.queries import (
    get_crude_birth_rate,
    get_death_rates_by_agegroup,
    get_iso3_from_country_name,
    get_life_expectancy_by_agegroup,
    get_population_by_agegroup,
)
from autumn.core.inputs.john_hopkins.fetch import fetch_john_hopkins_data
from autumn.core.inputs.john_hopkins.queries import get_john_hopkins_data
from autumn.core.inputs.mobility.fetch import fetch_mobility_data
from autumn.core.inputs.mobility.queries import get_mobility_data
from autumn.core.inputs.social_mixing.queries import get_country_mixing_matrix
from autumn.core.inputs.owid.fetch import fetch_owid_data

from autumn.coremer import Timer


def fetch_input_data():
    """
    Fetch input data from external sources,
    which can then be used to build the input database.
    """
    with Timer("Fetching mobility data."):
        fetch_mobility_data()

    with Timer("Fetching John Hopkins data."):
        fetch_john_hopkins_data()

    with Timer("Fetching COVID PHL data."):
        fetch_covid_phl_data()

    with Timer("Fetching COVID PHL data."):
        fetch_covid_mys_data()

    with Timer("Fetching OWID data."):
        fetch_owid_data()

    with Timer("Fetching COVID MMR data."):
        fetch_covid_mmr_data()

    with Timer("Fetching COVID survey data."):
        fetch_covid_survey_data()
