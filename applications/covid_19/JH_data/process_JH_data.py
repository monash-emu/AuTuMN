import pandas as pd
import os
from numpy import diff

def get_all_jh_countries():
    """
    Determine the list of available countries from the John Hopkins database
    :return: a list of country names
    """
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "covid_confirmed.csv")
    data = pd.read_csv(file_path)
    countries = data['Country/Region'].to_list()

    # remove duplicates
    countries = list(dict.fromkeys(countries))

    return countries


def read_john_hopkins_data_from_csv(variable="confirmed", country='Australia'):
    """
    Read John Hopkins data from previously generated csv files
    :param variable: one of "confirmed", "deaths", "recovered"
    :param country: country
    """
    filename = "covid_" + variable + ".csv"
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)

    data = pd.read_csv(path)
    data = data[data['Country/Region'] == country]

    # We need to collect the country-level data
    if data['Province/State'].isnull().any():  # when there is a single row for the whole country
        data = data[data['Province/State'].isnull()]

    data_series = []
    for (columnName, columnData) in data.iteritems():
        if columnName.count("/") > 1:
            cumul_this_day = sum(columnData.values)
            data_series.append(cumul_this_day)

    # for confirmed and deaths, we want the daily counts and not the cumulative number
    if variable != 'recovered':
        data_series = diff(data_series)
    return data_series
