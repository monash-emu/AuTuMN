import pandas as pd
import os
from numpy import diff, linspace

import matplotlib.pyplot as plt
from autumn.tool_kit.utils import find_first_index_reaching_cumulative_sum
from applications.covid_19.covid_model import *
from autumn.db import get_iso3_from_country_name

import yaml

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

    return data_series.tolist()


def plot_jh_data(data):
    """
    Produce a graph for each country
    :param data: a dictionary with the country names as keys and the data as values
    """
    dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_graphs')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    i = 0
    for country, n_cases in data.items():
        i += 1
        plt.figure(i)
        x = list(range(len(n_cases)))
        plt.bar(x, list(n_cases))
        filename = "daily_cases_" + country + ".png"
        path = os.path.join(dir_path, filename)
        plt.savefig(path)


def plot_fitted_model(country):
    dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_graphs')

    # load the calibrated parameters
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'mle_estimates',
                                             'covid_' + country, 'mle_params.yml'
                             )
    with open(file_path, 'r') as yaml_file:
        calibrated_params = yaml.safe_load(yaml_file)
    for param_name in calibrated_params:
        calibrated_params[param_name] = float(calibrated_params[param_name])
    calibrated_params['country'] = country
    calibrated_params['iso3'] = get_iso3_from_country_name(input_database, country)
    calibrated_params['end_time'] = 100
    # build the model
    this_model = build_covid_model(calibrated_params)
    this_model.run_model()

    model_times = this_model.derived_outputs['times']
    plot_times = [t - 22 for t in model_times]
    notifications = this_model.derived_outputs['notifications']

    # get the data
    n_daily_cases = read_john_hopkins_data_from_csv('confirmed', country=country)
    # get the subset of data points starting after 100th case detected and recording next 14 days
    index_100 = find_first_index_reaching_cumulative_sum(n_daily_cases, 100)
    data_of_interest = n_daily_cases[index_100: index_100 + 14]
    start_day = index_100 + 22  # because JH data starts 22/1

    plt.figure()
    x = list(range(len(n_daily_cases)))
    plt.bar(x, list(n_daily_cases), color='purple')
    x_2 = linspace(index_100, index_100+13, num=14)
    plt.bar(x_2, list(data_of_interest), color='magenta')

    plt.plot(plot_times, notifications, color='crimson', linewidth=2)
    plt.xlim((0, max(x)))
    plt.xlabel('days since 22/1/2020')
    plt.ylabel('daily notifications')
    plt.title(country)
    # plt.ylim((0, 1000))

    filename = "fitted_model_" + country + ".png"
    path = os.path.join(dir_path, filename)
    plt.savefig(path)

#
# for _country in ['Canada', 'Germany', 'France', 'Australia', 'Italy', 'China', 'Pakistan', 'Spain']:
#     plot_fitted_model(_country)
