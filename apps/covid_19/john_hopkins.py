"""
Access John Hopkins COVID-19 data from their GitHub repository.
"""
import os
import yaml
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
from numpy import diff, linspace
import matplotlib.pyplot as plt

from autumn.db import get_iso3_from_country_name
from autumn.tool_kit.utils import find_first_index_reaching_cumulative_sum
from autumn.constants import DATA_PATH

JH_DATA_DIR = os.path.join(DATA_PATH, "john-hopkins")
GITHUB_BASE_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"

country_mapping = {
    "united-kingdom": "United Kingdom"
}

def get_all_jh_countries():
    """
    Determine the list of available countries from the John Hopkins database
    :return: a list of country names
    """
    download_jh_data()
    file_path = os.path.join(JH_DATA_DIR, "covid_confirmed.csv")
    data = pd.read_csv(file_path)
    countries = data["Country/Region"].to_list()
    countries = list(dict.fromkeys(countries))
    return countries


def read_john_hopkins_data_from_csv(variable="confirmed", country="Australia"):
    """
    Read John Hopkins data from previously generated csv files
    :param variable: one of "confirmed", "deaths", "recovered"
    :param country: country
    """
    if country in country_mapping:
        country_name = country_mapping[country]
    else:
        country_name = country
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


def print_jh_data_series(variable_list=["confirmed", "deaths"], country="Australia"):
    start_time = 22
    for variable in variable_list:
        print(variable)
        data = read_john_hopkins_data_from_csv(variable, country)
        times = [start_time + i for i in range(len(data))]
        print("times:")
        print(times)
        print("list for calibration:")
        print(data)
        print("list for plotting targets:")
        print([[d] for d in data])
        print()


def plot_jh_data(data):
    """
    Produce a graph for each country
    :param data: a dictionary with the country names as keys and the data as values
    """
    download_jh_data()
    dir_path = os.path.join(JH_DATA_DIR, "data_graphs")
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
    download_jh_data()
    dir_path = os.path.join(JH_DATA_DIR, "data_graphs")

    # load the calibrated parameters
    file_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "mle_estimates",
        "covid_" + country,
        "mle_params.yml",
    )
    with open(file_path, "r") as yaml_file:
        calibrated_params = yaml.safe_load(yaml_file)
    for param_name in calibrated_params:
        calibrated_params[param_name] = float(calibrated_params[param_name])
    calibrated_params["country"] = country
    calibrated_params["iso3"] = get_iso3_from_country_name(input_database, country)
    calibrated_params["end_time"] = 100
    # build the model
    this_model = build_model(calibrated_params)
    this_model.run_model()

    model_times = this_model.derived_outputs["times"]
    plot_times = [t - 22 for t in model_times]
    notifications = this_model.derived_outputs["notifications"]

    # get the data
    n_daily_cases = read_john_hopkins_data_from_csv("confirmed", country=country)
    # get the subset of data points starting after 100th case detected and recording next 14 days
    index_100 = find_first_index_reaching_cumulative_sum(n_daily_cases, 100)
    data_of_interest = n_daily_cases[index_100 : index_100 + 14]
    start_day = index_100 + 22  # because JH data starts 22/1

    plt.figure()
    x = list(range(len(n_daily_cases)))
    plt.bar(x, list(n_daily_cases), color="purple")
    x_2 = linspace(index_100, index_100 + 13, num=14)
    plt.bar(x_2, list(data_of_interest), color="magenta")

    plt.plot(plot_times, notifications, color="crimson", linewidth=2)
    plt.xlim((0, max(x)))
    plt.xlabel("days since 22/1/2020")
    plt.ylabel("daily notifications")
    plt.title(country)
    # plt.ylim((0, 1000))

    filename = "fitted_model_" + country + ".png"
    path = os.path.join(dir_path, filename)
    plt.savefig(path)


CSVS_TO_READ = [
    [
        "who_situation_report.csv",
        "who_covid_19_situation_reports/who_covid_19_sit_rep_time_series/who_covid_19_sit_rep_time_series.csv",
    ],
    [
        "covid_confirmed.csv",
        "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
    ],
    [
        "covid_deaths.csv",
        "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
    ],
    [
        "covid_recovered.csv",
        "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv",
    ],
]


def download_jh_data():
    if not os.path.exists(JH_DATA_DIR):
        os.makedirs(JH_DATA_DIR)
    download_global_csv(JH_DATA_DIR)


def download_global_csv(output_dir: str):
    """
    Download John Hopkins COVID data as CSV to output_dir.
    """
    for filename, url_path in CSVS_TO_READ:
        url = urljoin(GITHUB_BASE_URL, url_path)
        path = os.path.join(output_dir, filename)
        df = pd.read_csv(url)
        df.to_csv(path)


def download_daily_reports(output_dir: str):
    dates = pd.date_range(start=datetime.today(), end=datetime(2020, 1, 22))
    for date in dates:
        filename = date.strftime("%m-%d-%Y.csv")
        url = urljoin(GITHUB_BASE_URL, "csse_covid_19_data/csse_covid_19_daily_reports", filename)
        path = os.path.join(output_dir, filename)
        df = pd.read_csv(url)
        df.to_csv(path)


# download_jh_data()
# print_jh_data_series(country="United Kingdom")
