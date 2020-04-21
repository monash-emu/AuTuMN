import os
from autumn import constants

from autumn.demography.social_mixing import get_all_prem_countries
from autumn.db import Database
from applications.covid_19.JH_data.process_JH_data import (
    get_all_jh_countries,
    read_john_hopkins_data_from_csv,
    plot_jh_data,
)

INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")

input_database = Database(database_name=INPUT_DB_PATH)

prem_country_list = get_all_prem_countries()  # N=152
jh_country_list = get_all_jh_countries()  # N=180
intercept_country_list = list(set(prem_country_list) & set(jh_country_list))  # N=126

all_data = {}
for i, country in enumerate(intercept_country_list):
    all_data[country] = read_john_hopkins_data_from_csv(country=country)
# plot_jh_data(all_data)

# print list of countries with more than 1000 cases
countries_1000 = []
for country, n_cases in all_data.items():
    if sum(n_cases) >= 1000:
        countries_1000.append(country)
print(countries_1000)
# ['Dominican Republic', 'Singapore', 'Qatar', 'Greece', 'Canada', 'Israel', 'Portugal', 'Brazil', 'India', 'Estonia', 'Denmark', 'Saudi Arabia', 'Germany', 'Iceland', 'Luxembourg', 'South Africa', 'Spain', 'Algeria', 'Argentina', 'Indonesia', 'Peru', 'Thailand', 'France', 'Poland', 'Philippines', 'Australia', 'Malaysia', 'Italy', 'Serbia', 'Japan', 'Ireland', 'Romania', 'Sweden', 'Egypt', 'Pakistan', 'Switzerland', 'Mexico', 'Netherlands', 'Morocco', 'Turkey', 'United Arab Emirates', 'Slovenia', 'Austria', 'Panama', 'New Zealand', 'China', 'Ukraine', 'Chile', 'Belgium', 'Finland', 'Croatia', 'Iraq', 'Ecuador', 'Colombia']
