import numpy as np

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.tools.calibration.targets import NormalTarget, PoissonTarget, TruncNormalTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS

CLUSTERS = [Region.to_filename(r) for r in Region.VICTORIA_SUBREGIONS]

# Just calibrate to June, July, August and September for now (but run for some lead in time at the start)
TARGETS_START_TIME = 153  # 1st June
TARGETS_END_TIME = 305  # 31st October
TARGETS_RANGE = (TARGETS_START_TIME, TARGETS_END_TIME)
DISPERSION_TARGET_RATIO = 0.07


# Load and configure model parameters
default_path = build_rel_path("params/default.yml")
mle_path = build_rel_path("params/mle-params.yml")
scenario_paths = [build_rel_path(f"params/scenario-{i}.yml") for i in range(1, 5)]
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Add calibration targets and priors
ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))

cluster_targets = []
for cluster in CLUSTERS:
    notifs_ts = ts_set.get(f"notifications_for_cluster_{cluster}").moving_average(4)
    dispersion_value = max(notifs_ts.values) * DISPERSION_TARGET_RATIO
    target = NormalTarget(notifs_ts, stdev=dispersion_value)
    cluster_targets.append(target)

targets = [
    PoissonTarget(ts_set.get("notifications").truncate_times(*TARGETS_RANGE).round_values()),
    PoissonTarget(ts_set.get("infection_deaths").truncate_times(*TARGETS_RANGE).moving_average(7)),
    PoissonTarget(ts_set.get("hospital_admissions").truncate_times(*TARGETS_RANGE)),
    PoissonTarget(ts_set.get("icu_admissions").truncate_times(*TARGETS_RANGE)),
    # FIXME: The target below may need to be included for the revised analysis
    # TruncNormalTarget(ts_set.get("prop_notifications_elderly"), trunc_range=[0., 1.], stdev=.1),
    *cluster_targets,
]


cluster_priors = []
# Add multiplier for each Victorian cluster
for region in Region.VICTORIA_METRO:
    region_name = region.replace("-", "_")
    name = f"victorian_clusters.contact_rate_multiplier_{region_name}"
    # Shouldn't be too peaked with these values.
    prior = TruncNormalPrior(name, mean=1.0, stdev=0.5, trunc_range=[0.5, np.inf])
    cluster_priors.append(prior)


priors = [
    # Global COVID priors.
    *COVID_GLOBAL_PRIORS,
    # Cluster specific priors.
    *cluster_priors,
    # Victorian regional priors.
    # Shouldn't be too peaked with these values
    TruncNormalPrior(
        f"victorian_clusters.contact_rate_multiplier_barwon_south_west",
        mean=1.0,
        stdev=0.5,
        trunc_range=[0.5, np.inf],
        jumping_stdev=0.05,
    ),
    # Shouldn't be too peaked with these values
    TruncNormalPrior(
        f"victorian_clusters.contact_rate_multiplier_regional",
        mean=1.0,
        stdev=0.5,
        trunc_range=[0.5, np.inf],
        jumping_stdev=0.05,
    ),
    UniformPrior(
        "contact_rate",
        [0.015, 0.06],
        jumping_stdev=0.002,
    ),
    UniformPrior(
        "victorian_clusters.intercluster_mixing",
        [0.005, 0.05],
        jumping_stdev=0.001,
    ),
    # Should be multiplied by 4/9 because seed is removed from regional clusters
    UniformPrior("infectious_seed", [22.5, 67.5], jumping_stdev=2.0),
    TruncNormalPrior(
        "clinical_stratification.props.symptomatic.multiplier",
        mean=1.0,
        stdev=0.2,
        trunc_range=[0.5, np.inf],
    ),
    UniformPrior(
        "clinical_stratification.non_sympt_infect_multiplier", [0.15, 0.7], jumping_stdev=0.01
    ),
    UniformPrior(
        "clinical_stratification.props.hospital.multiplier", [0.5, 3.0], jumping_stdev=0.1
    ),
    UniformPrior("infection_fatality.multiplier", [0.5, 4.0], jumping_stdev=0.05),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.2, 0.5], jumping_stdev=0.01),
    TruncNormalPrior(
        "sojourn.compartment_periods.icu_early",
        mean=12.7,
        stdev=4.0,
        trunc_range=[3.0, np.inf],
        jumping_stdev=2.0,
    ),
    UniformPrior(
        "victorian_clusters.metro.mobility.microdistancing.behaviour_adjuster.parameters.effect",
        [0.0, 0.5],
        jumping_stdev=0.005,
    ),
    UniformPrior(
        "victorian_clusters.metro.mobility.microdistancing.face_coverings_adjuster.parameters.effect",
        [0.0, 0.5],
        jumping_stdev=0.005,
    ),
    UniformPrior("target_output_ratio", [0.1, 0.4], jumping_stdev=0.005),
]
calibration = Calibration(
    priors,
    targets,
    metropolis_init="current_params",
    metropolis_init_rel_step_size=0.05,
    fixed_proposal_steps=500,
)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("targets.secret.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(
    Region.VICTORIA, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)

# Write parameter table to tex file
main_table_params_list = [
    "clinical_stratification.icu_prop",
    "sojourn.compartment_periods_calculated.exposed.total_period",
    "contact_rate"
]
# project.write_params_to_tex(main_table_params_list, project_path=build_rel_path(''))
