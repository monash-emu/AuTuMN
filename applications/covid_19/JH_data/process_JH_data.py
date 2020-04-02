import pandas as pd


def get_all_jh_countries():
    """
    Determine the list of available countries from the John Hopkins database
    :return: a list of country names
    """

    filename = "covid_confirmed.csv"
    data = pd.read_csv(filename)
    countries = data['Country/Region'].to_list()

    # remove duplicates
    countries = list(dict.fromkeys(countries))

    return countries
