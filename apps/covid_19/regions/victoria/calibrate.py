import numpy as np

from apps.covid_19.model.victorian_outputs import CLUSTERS
from apps.covid_19.calibration import (
    provide_default_calibration_params, get_truncated_output
)
from autumn.constants import Region
from autumn.tool_kit.params import load_targets
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from apps.covid_19 import calibration as base
from autumn.tool_kit.utils import apply_moving_average

CLUSTERS = [Region.to_filename(r) for r in Region.VICTORIA_SUBREGIONS]

# Just calibrate to June, July, August and September for now (but run for some lead in time at the start)
TARGETS_START_TIME = 153  # 1st June
TARGETS_END_TIME = 305  # 31st October


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    target_outputs = get_target_outputs(TARGETS_START_TIME, TARGETS_END_TIME)
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

    # Get common parameters for all Covid applications
    priors = provide_default_calibration_params()

    # Add multiplier for each Victorian cluster
    for region in Region.VICTORIA_SUBREGIONS:
        region_name = region.replace("-", "_")
        priors += [
            {
                "param_name": f"victorian_clusters.contact_rate_multiplier_{region_name}",
                "distribution": "trunc_normal",
                "distri_params": [1., 0.5],  # Shouldn't be too peaked with these values
                "trunc_range": [0.5, np.inf],
            },
        ]

    # Add the other parameters we're interested in for Victoria
    priors += [
        {
            "param_name": "contact_rate",
            "distribution": "uniform",
            "distri_params": [0.015, 0.06],
        },
        {
            "param_name": "infectious_seed",
            "distribution": "uniform",
            "distri_params": [20., 60.],
        },
        {
            "param_name": "seasonal_force",
            "distribution": "uniform",
            "distri_params": [0.0, 0.5],
        },
        {
            "param_name": "clinical_stratification.props.symptomatic.multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.1],
            "trunc_range": [0.5, np.inf],
        },
        {
            "param_name": "infection_fatality.multiplier",
            "distribution": "uniform",
            "distri_params": [0.5, 4.],
        },
        {
            "param_name": "testing_to_detection.assumed_cdr_parameter",
            "distribution": "uniform",
            "distri_params": [0.2, 0.5],
        },
        {
            "param_name": "clinical_stratification.props.hospital.multiplier",
            "distribution": "uniform",
            "distri_params": [0.5, 3.0],
        },
        {
            "param_name": "sojourn.compartment_periods.icu_early",
            "distribution": "trunc_normal",
            "distri_params": [12.7, 4.0],
            "trunc_range": [3.0, np.inf],
        },
        {
            "param_name": "sojourn.compartment_periods.icu_late",
            "distribution": "trunc_normal",
            "distri_params": [10.8, 4.0],
            "trunc_range": [6.0, np.inf],
        },
        {
            "param_name": "victorian_clusters.intercluster_mixing",
            "distribution": "uniform",
            "distri_params": [0.005, 0.05],
        },
        {
            "param_name": "victorian_clusters.metro.mobility.microdistancing.behaviour.parameters.upper_asymptote",
            "distribution": "uniform",
            "distri_params": [0., 0.5],
        },
        {
            "param_name": "victorian_clusters.metro.mobility.microdistancing.face_coverings.parameters.upper_asymptote",
            "distribution": "uniform",
            "distri_params": [0., 0.5],
        },
    ]

    priors = add_dispersion_param_prior_for_gaussian(priors, target_outputs)
    priors = group_dispersion_params(priors, target_outputs)
    return priors


def get_target_outputs(start_date, end_date):
    targets = load_targets("covid_19", Region.VICTORIA)

    # Total Victorian notifications for each time point
    notification_times, notification_values = \
        get_truncated_output(targets["notifications"], start_date, end_date)
    notification_values = [round(value) for value in notification_values]
    target_outputs = [
        {
            "output_key": "notifications",
            "years": notification_times,
            "values": notification_values,
            "loglikelihood_distri": "poisson",
        }
    ]

    targets["infection_deaths"]["times"], targets["infection_deaths"]["values"] = \
        get_truncated_output(targets["infection_deaths"], start_date, end_date)
    targets.update(base.accumulate_target(targets, "infection_deaths"))
    targets["accum_infection_deaths"]["values"] = \
        [round(i_value) for i_value in targets["accum_infection_deaths"]["values"]]
    target_outputs += [
        {
            "output_key": "accum_deaths",
            "years": [targets["accum_infection_deaths"]["times"][-1]],
            "values": [targets["accum_infection_deaths"]["values"][-1]],
            "loglikelihood_distri": "poisson",
        }
    ]

    hospitalisation_times, hospitalisation_values = \
        get_truncated_output(targets["hospital_admissions"], start_date, end_date)
    target_outputs += [
        {
            "output_key": "hospital_admissions",
            "years": hospitalisation_times,
            "values": hospitalisation_values,
            "loglikelihood_distri": "poisson",
        }
    ]

    # Smoothed notifications for all clusters
    for cluster in CLUSTERS:
        output_key = f"notifications_for_cluster_{cluster}"
        target_outputs += [
            {
                "output_key": output_key,
                "years": targets[output_key]["times"],
                "values": apply_moving_average(targets[output_key]["values"], period=4),
                "loglikelihood_distri": "normal",
            }
        ]

    return target_outputs


def group_dispersion_params(priors, target_outputs):
    """
    Reduce the number of fitted dispersion parameters. It will group all the dispersion parameters associated with a
    given output (e.g. 'notifications') and a given cluster type ('metro' or 'rural')
    :param priors: list of prior dictionaries
    :param target_outputs: list or target dictionaries
    :return: updated list of prior dictionaries
    """
    output_types = [
        "notifications",
        "hospital_admissions",
        "icu_admissions",
        "accum_deaths",
        "accum_notifications",
        "hospital_occupancy",
        "icu_occupancy",
    ]

    # remove all cluster-specific dispersion params that are no longer relevant
    removed_prior_idx = []
    for i, prior in enumerate(priors):
        if prior["param_name"].endswith("_dispersion_param") and "_for_cluster_" in prior["param_name"]:
            if prior["param_name"].split("_for_cluster_")[0] in output_types:
                removed_prior_idx.append(i)
    priors = [priors[j] for j in range(len(priors)) if j not in removed_prior_idx]

    # loop through all potential combinations of output type and cluster type
    clusters_by_group = {
        "metro": Region.VICTORIA_METRO,
        "rural": Region.VICTORIA_RURAL,
    }
    all_target_names = [t["output_key"] for t in target_outputs]
    for output_type in output_types:
        for cluster_type in ["metro", "rural"]:
            # test if this type of output for this cluster type is among the targets
            test_cluster = clusters_by_group[cluster_type][0].replace("-", "_")
            test_output_name = f"{output_type}_for_cluster_{test_cluster}"
            if test_output_name in all_target_names:

                # We need to create a new group dispersion parameter
                # First, we need to find the max value of the relevant targets to set an appropriate prior
                max_val = -1e6
                for t in target_outputs:
                    if "_for_cluster_" in t["output_key"]:
                        region_name = t['output_key'].split("_for_cluster_")[1]
                        if t['output_key'].startswith(output_type) and region_name.replace("_", "-") in clusters_by_group[cluster_type]:
                            assert t["loglikelihood_distri"] == "normal", \
                                "The dispersion parameter is designed for a Gaussian likelihood"
                            max_val = max(
                                max_val,
                                max(t["values"])
                            )

                # sd_ that would make the 95% gaussian CI cover half of the max value (4*sd = 95% width)
                sd_ = max_val * 0.25
                lower_sd = sd_ / 2.0
                upper_sd = 2.0 * sd_

                priors.append(
                    {
                        "param_name": f"{output_type}_{cluster_type}_dispersion_param",
                        "distribution": "uniform",
                        "distri_params": [lower_sd, upper_sd],
                    },
                )

    return priors

