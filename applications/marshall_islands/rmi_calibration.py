from autumn.calibration import Calibration

from applications.marshall_islands.rmi_model import build_rmi_model


def run_calibration_chain(max_seconds: int, run_id: int):
    """
    Run a calibration chain for the Marshall Islands TB model

    num_iters: Maximum number of iterations to run.
    available_time: Maximum time, in seconds, to run the calibration.
    """
    print(f"Preparing to run Marshall Islands TB model calibration for run {run_id}")
    calib = Calibration(
        "marshall_islands", build_rmi_model, PAR_PRIORS, TARGET_OUTPUTS, MULTIPLIERS, run_id
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
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.0001, 0.01]},
    {"param_name": "beta_decay_rate", "distribution": "uniform", "distri_params": [0.01, 0.1]},
    {
        "param_name": "minimum_tv_beta_multiplier",
        "distribution": "uniform",
        "distri_params": [0.1, 0.4],
    },
    {"param_name": "rr_transmission_ebeye", "distribution": "uniform", "distri_params": [1.0, 2.5]},
    {
        "param_name": "rr_transmission_otherislands",
        "distribution": "uniform",
        "distri_params": [0.5, 1.5],
    },
    {"param_name": "cdr_multiplier", "distribution": "uniform", "distri_params": [0.5, 2.0]},
    {
        "param_name": "case_detection_ebeye_multiplier",
        "distribution": "uniform",
        "distri_params": [0.5, 2.0],
    },
    {
        "param_name": "case_detection_otherislands_multiplier",
        "distribution": "uniform",
        "distri_params": [0.5, 1.0],
    },
    {
        "param_name": "over_reporting_prevalence_proportion",
        "distribution": "uniform",
        "distri_params": [0.0, 0.5],
    },
]

TARGET_OUTPUTS = [
    {
        "output_key": "prevXinfectiousXamongXlocation_ebeye",
        "years": [2017.0],
        "values": [755.0],
        "cis": [(620.0, 894.0)],
    },
    {
        "output_key": "reported_majuro_prevalence",
        "years": [2018.0],
        "values": [1578.0],
        "cis": [(620.0, 894.0)],
    },
    {
        "output_key": "prevXlatentXamongXlocation_majuro",
        "years": [2017.0],
        "values": [28.5],
        "cis": [(902.0, 1018.0)],
    },
    {"output_key": "notificationsXlocation_majuro", "years": [2016.0], "values": [119.0]},
    {"output_key": "notificationsXlocation_ebeye", "years": [2016.0], "values": [53.0]},
    {
        "output_key": "notificationsXlocation_otherislands",
        "years": [2016.0],
        "values": [6.0],
        "cis": [(5.0, 17.0)],
    },
]

MULTIPLIERS = {
    "prevXinfectiousXamong": 1.0e5,
    "prevXlatentXamongXlocation_majuro": 1.0e4,
    "prevXinfectiousXstrain_mdrXamongXinfectious": 1.0e4,
}

