"""
Read John Hopkins COVID-19 data.
"""
import pandas as pd
from numpy import diff

from .fetch import JH_DATA_DIR

COUNTRY_MAPPING = {"united-kingdom": "United Kingdom"}


def get_john_hopkins_data(variable: str, country: str, latest=False):
    """
    Read John Hopkins data from previously generated csv files
    :param variable: one of "confirmed", "deaths", "recovered"
    :param country: country
    """
    if country in COUNTRY_MAPPING:
        country_name = COUNTRY_MAPPING[country]
    else:
        country_name = country.title()
    download_jh_data()
    filename = f"covid_{variable}.csv"
    path = os.path.join(JH_DATA_DIR, filename)
    data = pd.read_csv(path)
    data = data[data["Country/Region"] == country_name]

    # We need to collect the country-level data
    if data["Province/State"].isnull().any():  # when there is a single row for the whole country
        data = data[data["Province/State"].isnull()]

    data_series = []
    for (columnName, columnData) in data.iteritems():
        if columnName.count("/") > 1:
            cumul_this_day = sum(columnData.values)
            data_series.append(cumul_this_day)

    # for confirmed and deaths, we want the daily counts and not the cumulative number
    if variable != "recovered":
        data_series = diff(data_series)

    return data_series.tolist()
