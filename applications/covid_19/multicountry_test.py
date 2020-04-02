import os
from autumn import constants

from autumn.demography.social_mixing import get_all_prem_countries
from autumn.db import Database, get_iso3_from_country_name

INPUT_DB_PATH = os.path.join(constants.DATA_PATH, 'inputs.db')

input_database = Database(database_name=INPUT_DB_PATH)

country_list = get_all_prem_countries()

for i, country in enumerate(country_list):
    iso3 = get_iso3_from_country_name(input_database, country)
    print(iso3)

