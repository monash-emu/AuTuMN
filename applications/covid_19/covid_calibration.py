from autumn.calibration import Calibration
from autumn.tool_kit.utils import find_first_index_reaching_cumulative_sum

from applications.covid_19.covid_model import build_covid_model, PARAMS_PATH
from applications.covid_19.JH_data.process_JH_data import read_john_hopkins_data_from_csv

from numpy import linspace
import yaml

with open(PARAMS_PATH, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
scenario_params = params['scenarios']
sc_start_time = params['scenario_start']


def run_calibration_chain(max_seconds: int, run_id: int):
    """
    Run a calibration chain for the covid model

    num_iters: Maximum number of iterations to run.
    available_time: Maximum time, in seconds, to run the calibration.
    """
    print(f"Preparing to run covid model calibration for run {run_id}")
    calib = Calibration(
        "covid", build_covid_model, PAR_PRIORS, TARGET_OUTPUTS, MULTIPLIERS, run_id,
        scenario_params, sc_start_time, model_parameters=params['default']
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

# for JH data, day_1 is '1/22/20', that is 22 Jan 2020
n_daily_cases = read_john_hopkins_data_from_csv('confirmed', country=params['default']['country'])

# get the subset of data points starting after 100th case detected and recording next 14 days
index_100 = find_first_index_reaching_cumulative_sum(n_daily_cases, 100)
data_of_interest = n_daily_cases[index_100: index_100 + 14]

start_day = index_100 + 22  # because JH data starts 22/1


PAR_PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [.3, .6]},
    {"param_name": "start_time", "distribution": "uniform", "distri_params": [-30, start_day - 1]}
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": linspace(start_day, start_day + len(data_of_interest) - 1, num=len(data_of_interest)),
        "values": data_of_interest,
        "loglikelihood_distri": 'poisson'
    }
]

MULTIPLIERS = {

}

run_calibration_chain(120, 1)
