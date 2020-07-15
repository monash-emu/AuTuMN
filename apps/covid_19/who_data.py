import os
import pandas as pd
from autumn.constants import DATA_PATH
import datetime


WHO_DATA_DIR = os.path.join(DATA_PATH, "who_covid")
country_mapping = {"united-kingdom": "The United Kingdom"}


def read_who_data_from_csv(variable="confirmed", country="Australia", data_start_time=50):
    """
    Read WHO data from csv files
    :param variable: one of "confirmed", "deaths"
    :param country: country
    """
    if country in country_mapping:
        country_name = country_mapping[country]
    else:
        country_name = country.title()
    filename = f"WHO-COVID-19-global-data.csv"
    path = os.path.join(WHO_DATA_DIR, filename)
    data = pd.read_csv(path)
    times = list(data[data.iloc[:, 2] == country_name].iloc[:, 0])

    date_ref = datetime.date(2019, 12, 31)
    for i, time_string in enumerate(times):
        components = time_string.split("-")
        time_date = datetime.date(int(components[0]), int(components[1]), int(components[2]))
        delta = (time_date - date_ref)
        times[i] = delta.days

    start_index = times.index(data_start_time)

    times = times[start_index:]
    if variable == 'confirmed':
        values = list(data[data.iloc[:, 2] == country_name].iloc[:, 4])[start_index:]
    elif variable == 'deaths':
        values = list(data[data.iloc[:, 2] == country_name].iloc[:, 6])[start_index:]

    return times, values
