import pandas as pd
import os

def get_all_jh_countries():
    """
    Determine the list of available countries from the John Hopkins database
    :return: a list of country names
    """
    filepath=os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "covid_confirmed.csv")
    data = pd.read_csv(filepath)
    countries = data['Country/Region'].to_list()

    # remove duplicates
    countries = list(dict.fromkeys(countries))

    return countries
