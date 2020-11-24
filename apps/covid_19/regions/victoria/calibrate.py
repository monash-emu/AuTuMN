from apps.covid_19.model.victorian_outputs import CLUSTERS
from apps.covid_19.calibration import (
    add_standard_victoria_params,
    provide_default_calibration_params,
)
from autumn.constants import Region
from autumn.tool_kit.params import load_targets
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from apps.covid_19 import calibration as base

IRRELEVANT_PRIORS = ["importation.movement_prop"]
CLUSTERS = [Region.to_filename(r) for r in Region.VICTORIA_SUBREGIONS]


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    target_outputs = get_target_outputs()
    priors = get_priors(target_outputs)
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.VICTORIA,
        priors,
        target_outputs,
    )


def get_priors(target_outputs: list):
    priors = provide_default_calibration_params(("time.start", "contact_rate"))
    priors = add_standard_victoria_params(priors, Region.VICTORIA)
    priors = add_dispersion_param_prior_for_gaussian(priors, target_outputs)
    priors = [p for p in priors if not p["param_name"] in IRRELEVANT_PRIORS]
    return priors


def get_target_outputs():
    targets = load_targets("covid_19", Region.VICTORIA)
    target_outputs = []

    # Calibrate all Victoria sub-regions to notifications
    for cluster in CLUSTERS:
        ignore_last = 2
        output_key = f"notifications_for_cluster_{cluster}"
        notification_times = targets[output_key]["times"][:-ignore_last]
        notification_values = targets[output_key]["values"][:-ignore_last]

        target_outputs += [
            {
                "output_key": targets[output_key]["output_key"],
                "years": notification_times,
                "values": notification_values,
                "loglikelihood_distri": "normal",
            }
        ]

    return target_outputs