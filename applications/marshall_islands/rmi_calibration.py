from autumn.calibration import Calibration

from applications.marshall_islands.rmi_model import build_rmi_model, PARAMS_PATH

import yaml

with open(PARAMS_PATH, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)


def run_calibration_chain(max_seconds: int, run_id: int):
    """
    Run a calibration chain for the Marshall Islands TB model

    num_iters: Maximum number of iterations to run.
    available_time: Maximum time, in seconds, to run the calibration.
    """
    print(f"Preparing to run Marshall Islands TB model calibration for run {run_id}")
    calib = Calibration(
        "marshall_islands", build_rmi_model, PAR_PRIORS, TARGET_OUTPUTS, MULTIPLIERS, run_id,
        model_parameters=params['default']
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
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [10., 20.]},
    {
        "param_name": "late_progression_15",
        "distribution": "lognormal",
        "distri_params": [-12.11, 0.45]},  # Ragonnet et al, Epidemics 2017
    {
        "param_name": "rr_progression_diabetic",
        "distribution": "uniform",
        "distri_params": [2.25, 5.73]},
    {
        "param_name": "rr_transmission_ebeye",
        "distribution": "lognormal",
        "distri_params": [.25, .5]},
    {
        "param_name": "rr_transmission_otherislands",
        "distribution": "lognormal",
        "distri_params": [.25, .5]},
    {
        "param_name": "cdr_multiplier",
        "distribution": "uniform",
        "distri_params": [0.5, 2.0]},
    {
        "param_name": "case_detection_ebeye_multiplier",
        "distribution": "lognormal",
        "distri_params": [.25, .5]},
    {
        "param_name": "case_detection_otherislands_multiplier",
        "distribution": "lognormal",
        "distri_params": [.25, .5]},
    {
        "param_name": "over_reporting_prevalence_proportion",
        "distribution": "uniform",
        "distri_params": [0.0, 0.5],},
]

TARGET_OUTPUTS = [
    {
        "output_key": "prevXinfectiousXamongXlocation_ebeye",
        "years": [2017.0],
        "values": [755.0],
        "cis": [(620.0, 894.0)],},
    {
        "output_key": "reported_majuro_prevalence",
        "years": [2018.0],
        "values": [1578.0],},
    {
        "output_key": "prevXlatentXamongXlocation_majuro",
        "years": [2018.0],
        "values": [28.5],},
    {
        "output_key": "notificationsXlocation_majuro",
        "years": [2016.0],
        "values": [119.0]},
    {
        "output_key": "notificationsXlocation_ebeye",
        "years": [2016.0],
        "values": [53.0]},
    {
        "output_key": "notificationsXlocation_otherislands",
        "years": [2014.0],
        "values": [10.0]
    },
]

MULTIPLIERS = {
    "prevXinfectiousXamong": 1.0e5,
    "prevXinfectiousXamongXlocation_ebeye": 1.0e5,
    "prevXlatentXamongXlocation_majuro": 100
}
