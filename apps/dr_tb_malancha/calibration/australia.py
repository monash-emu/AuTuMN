from autumn.constants import Region
from apps.dr_tb_malancha.calibration import base


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.AUSTRALIA,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
        _multipliers=MULTIPLIERS,
    )


PAR_PRIORS = [
    {"param_name": "beta", "distribution": "uniform", "distri_params": [1.0, 4.0],},
]

MULTIPLIERS = {"prevXinfectiousXamong": 100000}

TARGET_OUTPUTS = [
    {
        "output_key": "prevXinfectiousXamong",
        "years": [2000],
        "values": [200],
        "loglikelihood_distri": "normal",
    },
]

if __name__ == "__main__":
    run_calibration_chain(5, 0)
