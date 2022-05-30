"""
Fetch John Hopkins COVID-19 data from their GitHub repository.
"""
import os
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd

from autumn.settings import INPUT_DATA_PATH

JH_DATA_DIR = os.path.join(INPUT_DATA_PATH, "john-hopkins")
GITHUB_BASE_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
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


def fetch_john_hopkins_data():
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


# Not currently used.
def download_daily_reports(output_dir: str):
    dates = pd.date_range(start=datetime.today(), end=datetime(2020, 1, 22))
    for date in dates:
        filename = date.strftime("%m-%d-%Y.csv")
        url = urljoin(GITHUB_BASE_URL, "csse_covid_19_data/csse_covid_19_daily_reports", filename)
        path = os.path.join(output_dir, filename)
        df = pd.read_csv(url)
        df.to_csv(path)
