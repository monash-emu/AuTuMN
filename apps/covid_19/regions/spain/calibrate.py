from apps.covid_19 import calibration as base
from autumn.constants import Region
from apps.covid_19.mixing_optimisation.utils import (
    get_prior_distributions_for_opti,
    get_target_outputs_for_opti,
    get_weekly_summed_targets,
    add_dispersion_param_prior_for_gaussian,
)
from autumn.tool_kit.utils import print_target_to_plots_from_calibration


country = Region.SPAIN

# START CALIBRATION VALUES
# END CALIBRATION VALUES

PAR_PRIORS = get_prior_distributions_for_opti()
TARGET_OUTPUTS = get_target_outputs_for_opti(
    country, source="who", data_start_time=61, data_end_time=182
)

# Use weekly counts
for target in TARGET_OUTPUTS:
    target["years"], target["values"] = get_weekly_summed_targets(target["years"], target["values"])


# Add target for Prop immune
TARGET_OUTPUTS.append(
    {
        "output_key": "proportion_seropositive",
        "years": [125.0],
        "values": [0.05],
        "sd": 0.004,
        "loglikelihood_distri": "normal",
    }
)

PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)

# par_grid = [
#     {"param_name": "contact_rate", "lower": 0.05197245019634231, "upper": .0542, "n": 2},
#     {"param_name": "time_variant_detection.end_value", "lower":  0.3, "upper": 0.6446446106435421, "n": 2},
#     {"param_name": "notifications_dispersion_param", "lower":  2.612847051667734, "upper": 2.612847051667734, "n": 1},
#     {"param_name": "infection_deathsXall_dispersion_param", "lower":  0.35446059129060864, "upper": 0.35446059129060864, "n": 1},
#
# ]


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds, run_id, num_chains, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc"
    )


if __name__ == "__main__":
    print_target_to_plots_from_calibration(TARGET_OUTPUTS)
