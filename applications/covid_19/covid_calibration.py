from autumn.calibration import Calibration

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


PAR_PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [.3, .6]},
    # {"param_name": "start_time", "distribution": "uniform", "distri_params": [0., 65.]}
]
# for JH data, day_1 is '1/22/20', that is 22 Jan 2020
n_daily_cases = read_john_hopkins_data_from_csv('confirmed', country=params['default']['country'])

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": linspace(22, 21 + len(n_daily_cases), num=len(n_daily_cases))[-15:],
        "values": n_daily_cases[-15:]
    }
]

MULTIPLIERS = {

}
