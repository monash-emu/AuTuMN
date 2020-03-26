from autumn.calibration import Calibration

from applications.covid_19.covid_model import build_covid_model

import pandas as pd
import os
from numpy import diff, linspace


def run_calibration_chain(max_seconds: int, run_id: int):
    """
    Run a calibration chain for the covid model

    num_iters: Maximum number of iterations to run.
    available_time: Maximum time, in seconds, to run the calibration.
    """
    print(f"Preparing to run covid model calibration for run {run_id}")
    calib = Calibration(
        "covid", build_covid_model, PAR_PRIORS, TARGET_OUTPUTS, MULTIPLIERS, run_id
    )
    print("Starting calibration.")
    calib.run_fitting_algorithm(
        run_mode="autumn_mcmc",
        n_iterations=100000,
        n_burned=0,
        n_chains=1,
        available_time=max_seconds,
    )
    print(f"Finished calibration for run {run_id}.")


PAR_PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.3, 1.]},
    {"param_name": "start_time", "distribution": "uniform", "distri_params": [0., 30.]}
]


def read_john_hopkins_data_from_csv(variable="confirmed", country="Australia"):
    """
    Read John Hopkins data from previously generated csv files
    :param variable: one of "confirmed", "deaths", "recovered"
    :param country: country
    """
    filename = "covid_" + variable + ".csv"
    path = os.path.join('applications', 'covid_19', 'JH_data', filename)
    data = pd.read_csv(path)
    data = data[data['Country/Region'] == country]
    data_series = []
    for (columnName, columnData) in data.iteritems():
        if columnName.count("/") > 1:
            cumul_this_day = sum(columnData.values)
            data_series.append(cumul_this_day)

    # for confirmed and deaths, we want the daily counts and not the cumulative number
    if variable != 'recovered':
        data_series = diff(data_series)
    return data_series

# for JH data, day_1 is '1/22/20', that is 22 Jan 2020


n_daily_cases = read_john_hopkins_data_from_csv('confirmed')


TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": linspace(22, 21 + len(n_daily_cases), num=len(n_daily_cases))[-15:],
        "values": n_daily_cases[-15:]
    }
]

print(TARGET_OUTPUTS)

MULTIPLIERS = {

}
