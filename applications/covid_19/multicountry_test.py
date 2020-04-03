import os
from autumn import constants

from autumn.demography.social_mixing import get_all_prem_countries
from autumn.db import Database, get_iso3_from_country_name
from applications.covid_19.JH_data.process_JH_data import get_all_jh_countries, read_john_hopkins_data_from_csv, plot_jh_data

INPUT_DB_PATH = os.path.join(constants.DATA_PATH, 'inputs.db')

input_database = Database(database_name=INPUT_DB_PATH)

prem_country_list = get_all_prem_countries()  # N=152
jh_country_list = get_all_jh_countries()  # N=180
intercept_country_list = list(set(prem_country_list) & set(jh_country_list))  # N=126

all_data = {}
for i, country in enumerate(intercept_country_list):
    all_data[country] = read_john_hopkins_data_from_csv(country=country)
plot_jh_data(all_data)
