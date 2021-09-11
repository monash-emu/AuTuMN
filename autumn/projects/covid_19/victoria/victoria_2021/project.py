import numpy as np

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, use_tuned_proposal_sds
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.tools.calibration.targets import NormalTarget, PoissonTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models


# TODO: Get calibration running
# TODO: Implement future vaccination roll-out - understand how to specify projected roll-out in our code
# TODO: Implement future vaccination roll-out - understand the future projections provided by Vida
# TODO: Check YouGov inputs to micro-distancing functions
# TODO: Apply David's marginal posteriors as priors code
# TODO: Possibly move testing function out to a more obvious point in the code (less important)
# DONE: Check what's happening with testing numbers
# DONE: Increase severity for Delta
# DONE: Make vaccination apply to adults only
# DONE: Turn off top bracket IFR overwrite
# DONE: Check how case detection is working
# DONE: Make sure vaccination coverage can't exceed 100%
# DONE: Check waning immunity is off to make things a bit simpler
# DONE: Checked multi-strain functionality is turned off


CLUSTERS = [Region.to_filename(r) for r in Region.VICTORIA_SUBREGIONS]

# Load and configure model parameters
default_path = build_rel_path("params/default.yml")
mle_path = build_rel_path("params/mle-params.yml")
scenario_dir_path = build_rel_path("params/")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
param_set = ParameterSet(baseline=baseline_params)

# Add calibration targets and priors
ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))
target_start_time = 600

# For all the cluster targets, a universal calibrated parameter called "target_output_ratio" is used to scale the
# dispersion parameter of the targets' normal likelihoods.
cluster_targets = []
for cluster in CLUSTERS:
    notifs_ts = ts_set.get(f"notifications_for_cluster_{cluster}").truncate_start_time(target_start_time).moving_average(4)
    target = NormalTarget(notifs_ts)
    cluster_targets.append(target)

# Request calibration targets
targets = [
    PoissonTarget(ts_set.get("notifications").round_values().truncate_start_time(target_start_time)),
    # PoissonTarget(ts_set.get("infection_deaths").moving_average(7).truncate_start_time(target_start_time)),
    PoissonTarget(ts_set.get("hospital_admissions").truncate_start_time(target_start_time)),
    PoissonTarget(ts_set.get("icu_admissions").truncate_start_time(target_start_time)),
    *cluster_targets,
]

# Add multiplier for most services, except use South Metro for South East Metro, use North Metro for West Metro
cluster_priors = []
regions_for_multipliers = [
    reg for reg in Region.VICTORIA_METRO if reg not in (Region.SOUTH_EAST_METRO, Region.WEST_METRO)
]
regions_for_multipliers.append(Region.BARWON_SOUTH_WEST)

for region in regions_for_multipliers:
    region_name = region.replace("-", "_")
    name = f"victorian_clusters.contact_rate_multiplier_{region_name}"
    # Shouldn't be too peaked with these values.
    prior = TruncNormalPrior(name, mean=1.0, stdev=0.5, trunc_range=[0.5, np.inf], jumping_stdev=0.15)
    cluster_priors.append(prior)

priors = [
    # Global COVID priors, but with jumping sds adjusted
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.exposed.total_period",
        mean=5.5,
        stdev=0.97,
        trunc_range=[1.0, np.inf],
        jumping_stdev=0.5,
    ),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.active.total_period",
        mean=6.5,
        stdev=0.77,
        trunc_range=[4.0, np.inf],
        jumping_stdev=0.4,
    ),
    # Cluster specific priors.
    *cluster_priors,
    # Victorian regional priors.
    TruncNormalPrior(
        f"victorian_clusters.contact_rate_multiplier_regional",
        mean=1.0,
        stdev=0.5,
        trunc_range=[0.5, np.inf],
        jumping_stdev=0.15,
    ),
    UniformPrior("contact_rate", [0.04, 0.07], jumping_stdev=0.008),
    UniformPrior("victorian_clusters.intercluster_mixing", [0.005, 0.05], jumping_stdev=0.01),
    UniformPrior("infectious_seed", [20., 70.], jumping_stdev=4.),
    UniformPrior("clinical_stratification.non_sympt_infect_multiplier", [0.2, 0.8], jumping_stdev=0.05),
    UniformPrior("clinical_stratification.props.hospital.multiplier", [0.5, 5.0], jumping_stdev=0.4),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.05, 0.3], jumping_stdev=0.04),
    TruncNormalPrior(
        "sojourn.compartment_periods.icu_early",
        mean=12.7,
        stdev=4.0,
        trunc_range=[5.0, np.inf],
        jumping_stdev=4.
    ),
    UniformPrior(
        "victorian_clusters.metro.mobility.microdistancing.behaviour_adjuster.parameters.effect",
        [0.0, 0.6],
        jumping_stdev=0.075,
    ),
    UniformPrior(
        "victorian_clusters.metro.mobility.microdistancing.face_coverings_adjuster.parameters.effect",
        [0.0, 0.6],
        jumping_stdev=0.04,
    ),
    UniformPrior(
        "victorian_clusters.metro.mobility.microdistancing.home_reduction.parameters.effect",
        [0.0, 0.4],
        jumping_stdev=0.04,
    ),
    UniformPrior("target_output_ratio", [0.2, 0.7], jumping_stdev=0.04),
    UniformPrior("contact_tracing.assumed_trace_prop", [0.2, 0.5], jumping_stdev=0.04),
]

# Load proposal sds from yml file
use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

calibration = Calibration(
    priors,
    targets,
    metropolis_init="current_params",
    metropolis_init_rel_step_size=0.05,
    fixed_proposal_steps=500,
    jumping_stdev_adjustment=0.8,
)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("targets.secret.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(
    Region.VICTORIA_2021, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
