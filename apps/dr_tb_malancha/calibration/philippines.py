from autumn.constants import Region
from apps.dr_tb_malancha.calibration import base


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.PHILIPPINES,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
        _multipliers=MULTIPLIERS,
    )


PAR_PRIORS = [
    {"param_name": "beta", "distribution": "uniform", "distri_params": [4.0, 5.0]},
    {"param_name": "epsilon", "distribution": "lognormal", "distri_params": [-6.78, 0.15]},
    {"param_name": "kappa", "distribution": "lognormal", "distri_params": [-4.50, 0.13]},
    {"param_name": "nu", "distribution": "lognormal", "distri_params": [-11.99, 0.34]},
    {"param_name": "gamma", "distribution": "gamma", "distri_mean": 0.2, "distri_ci": [0.16, 0.29]},
    {
        "param_name": "infect_death",
        "distribution": "gamma",
        "distri_mean": 0.054,
        "distri_ci": [0.05, 0.06],
    },
    # {"param_name": "universal_death_rateX", "distribution": "gamma", "distri_mean": .0142, "distri_ci": [0.0135, 0.015]},
    {
        "param_name": "prop_of_failures_developing_inh_R",
        "distribution": "uniform",
        "distri_params": [0.01, 0.2],
    },
    {
        "param_name": "prop_of_failures_developing_rif_R",
        "distribution": "uniform",
        "distri_params": [0.0005, 0.005],
    },
]

MULTIPLIERS = {
    "prevXinfectiousXamong": 100000,
    "prevXinfectiousXstrain_inh_RXamongXinfectious": 100,
    "prevXinfectiousXstrain_rif_RXamongXinfectious": 100,
    "prevXinfectiousXstrain_mdrXamongXinfectious": 100,
}

TARGET_OUTPUTS = [
    {
        "output_key": "prevXinfectiousXamong",
        "years": [2007, 2016],
        "values": [463, 512],
        "loglikelihood_distri": "normal",
    },
    {
        "output_key": "prevXinfectiousXstrain_inh_RXamongXinfectious",
        "years": [2007, 2016],
        "values": [6.1, 10.26],
        "loglikelihood_distri": "normal",
    },
    {
        "output_key": "prevXinfectiousXstrain_rif_RXamongXinfectious",
        "years": [2007, 2016],
        "values": [0.8, 0.61],
        "loglikelihood_distri": "normal",
    },
    {
        "output_key": "prevXinfectiousXstrain_mdrXamongXinfectious",
        "years": [2007, 2016],
        "values": [1.5, 1.39],
        "loglikelihood_distri": "normal",
    },
]

if __name__ == "__main__":
    run_calibration_chain(1000, 0)
