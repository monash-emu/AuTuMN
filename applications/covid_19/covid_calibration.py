from autumn.calibration import Calibration

from applications.covid_19.covid_model import build_covid_model


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
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.5, 0.7]},
]

TARGET_OUTPUTS = [
    {
        "output_key": "prevXinfectiousXamong",
        "years": [100.0],
        "values": [.15],
        "cis": [(.10, .20)],
    }
]

MULTIPLIERS = {

}
