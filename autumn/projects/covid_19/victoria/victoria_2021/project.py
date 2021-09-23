import numpy as np

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, use_tuned_proposal_sds
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.tools.calibration.targets import NormalTarget, PoissonTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models


# TODO: Check YouGov inputs to micro-distancing functions (face coverings and physical distancing)
#  - need to get Mili to do this, data at https://github.com/YouGov-Data/covid-19-tracker/blob/master/data/australia.zip
# TODO: Calibrate to statewide deaths
# TODO: Write everything up as a policy brief
# TODO: Write the methods up as a technical document
# TODO: Implement age and cluster-specific vaccination coverage functions (may just be too much complexity)
# TODO: Implement the roadmap, including school closures and mobility changes

# Note I have changed this to the Metro clusters only - unlike in the Victoria 2020 analysis
metro_clusters = [Region.to_filename(r) for r in Region.VICTORIA_SUBREGIONS]

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
for cluster in metro_clusters:
    notifs_ts = ts_set.get(
        f"notificationsXcluster_{cluster}"
    ).truncate_start_time(target_start_time).moving_average(4)
    target = NormalTarget(notifs_ts)
    cluster_targets.append(target)

# Request calibration targets
targets = [
    PoissonTarget(ts_set.get("notifications").round_values().truncate_start_time(target_start_time)),
    PoissonTarget(ts_set.get("hospital_admissions").truncate_start_time(target_start_time)),
    PoissonTarget(ts_set.get("icu_admissions").truncate_start_time(target_start_time)),
    *cluster_targets,
]

# Hacky way to emphasise the last time point
last_notification_time = targets[0].timeseries.times[-1]
targets.append(
    PoissonTarget(ts_set.get("notifications").round_values().truncate_start_time(last_notification_time - 1))
)

# Add multiplier for most services, except use South Metro for South East Metro, use North Metro for West Metro
cluster_priors = []
regions_for_multipliers = [
    reg for reg in Region.VICTORIA_METRO if reg not in (Region.SOUTH_EAST_METRO, Region.WEST_METRO)
]
regions_for_multipliers.append(Region.BARWON_SOUTH_WEST)

for region in regions_for_multipliers:
    region_name = region.replace("-", "_")
    name = f"victorian_clusters.contact_rate_multiplier_{region_name}"
    # Shouldn't be too peaked with these values
    prior = TruncNormalPrior(
        name,
        mean=1.0, stdev=0.5, trunc_range=[0.5, np.inf], jumping_stdev=0.15
    )
    cluster_priors.append(prior)

# Marginal distributions of Vic 2020 to consider as priors for Vic 2021
# "victorian_clusters.contact_rate_multiplier_regional", norm (0.7070792993624084, 0.11538988453463195)
# "sojourn.compartment_periods_calculated.exposed.total_period", norm (6.095798813756773, 0.7810560402997285)
# "sojourn.compartment_periods_calculated.active.total_period", norm (6.431724510638751, 0.6588899585941116)
# "victorian_clusters.contact_rate_multiplier_regional", norm (0.7070792993624084, 0.11538988453463195)
# "sojourn.compartment_periods.icu_early", norm (13.189283389438017, 3.267836334270357)
# "victorian_clusters.metro.mobility.microdistancing.behaviour_adjuster.parameters.effect",
#  norm (0.3336881545907932, 0.12974271665347392)
#  or beta (2.233261027002466, 1.7150557025357558, 0.00300823791519224, 0.5850818483284497)
# "victorian_clusters.metro.mobility.microdistancing.face_coverings_adjuster.parameters.effect",
#  norm (0.4590192843551404, 0.054643498605008924)
#  or beta (2.233261027002466, 1.7150557025357558, 0.00300823791519224, 0.5850818483284497)
# "contact_rate", updated (0.005097283966437761, 0.04484184883556176)
# "clinical_stratification.non_sympt_infect_multiplier",
#  beta (5.070057160691058, 2.0783831204948724, -0.04627612686595504, 0.8467253773323684)
# "clinical_stratification.props.hospital.multiplier", norm (3.072957401469314, 0.9230093569298286)
# "testing_to_detection.assumed_cdr_parameter", norm (0.1875980041535647, 0.05487574154515127)
# "clinical_stratification.icu_prop",
#  norm (0.1875980041535647, 0.05487574154515127)
#  or beta (2.1990413238757105, 1.8012738113610243, 0.05194745495028011, 0.24786655960440956)
# "target_output_ratio",
#  beta (2.3143351886463726, 1.0958870124857243, 0.19372944320390947, 0.5061375024454435)
#  or norm (0.5376752874675825, 0.11298858887538074)
# "contact_tracing.assumed_trace_prop", uniform (0.20052289494754472, 0.29896766288137805)

priors = [
    # Cluster specific priors
    *cluster_priors,

    # Global COVID priors, but with jumping sds adjusted
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.exposed.total_period",
        mean=6.095798813756773, stdev=0.7810560402997285, trunc_range=(1.0, np.inf), jumping_stdev=0.5
    ),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.active.total_period",
        mean=6.431724510638751, stdev=0.6588899585941116, trunc_range=(3.0, np.inf), jumping_stdev=0.4
    ),

    # Victorian regional priors
    TruncNormalPrior(
        "victorian_clusters.contact_rate_multiplier_regional",
        mean=0.7070792993624084, stdev=0.11538988453463195, trunc_range=(0.5, np.inf), jumping_stdev=0.15
    ),
    UniformPrior(
        "contact_rate",
        (0.1, 0.28), jumping_stdev=0.008
    ),
    UniformPrior(
        "victorian_clusters.intercluster_mixing",
        (0.005, 0.05), jumping_stdev=0.01
    ),
    UniformPrior(
        "clinical_stratification.non_sympt_infect_multiplier",
        (0.2, 0.8), jumping_stdev=0.05
    ),
    TruncNormalPrior(
        "clinical_stratification.props.hospital.multiplier",
        mean=3.072957401469314, stdev=0.9230093569298286, trunc_range=(0.5, np.inf), jumping_stdev=0.4
    ),
    UniformPrior(
        "testing_to_detection.assumed_cdr_parameter",
        (0.02, 0.15), jumping_stdev=0.04
    ),
    UniformPrior(
        "clinical_stratification.icu_prop",
        (0.15, 0.3), jumping_stdev=0.05
    ),
    TruncNormalPrior(
        "sojourn.compartment_periods.icu_early",
        mean=13.189283389438017, stdev=3.267836334270357, trunc_range=(5.0, np.inf), jumping_stdev=4.
    ),
    TruncNormalPrior(
        "victorian_clusters.metro.mobility.microdistancing.behaviour_adjuster.parameters.effect",
        mean=0.3336881545907932, stdev=0.12974271665347392, trunc_range=(0., 1.), jumping_stdev=0.075
    ),
    TruncNormalPrior(
        "victorian_clusters.metro.mobility.microdistancing.face_coverings_adjuster.parameters.effect",
        mean=0.4590192843551404, stdev=0.054643498605008924, trunc_range=(0., 1.), jumping_stdev=0.04
    ),
    UniformPrior(
        "victorian_clusters.metro.mobility.microdistancing.home_reduction.parameters.effect",
        (0.0, 0.4), jumping_stdev=0.04
    ),
    UniformPrior(
        "target_output_ratio",
        (0.2, 0.7), jumping_stdev=0.04
    ),
    UniformPrior(
        "contact_tracing.assumed_trace_prop",
        (0.35, 0.6), jumping_stdev=0.04
    ),
    UniformPrior(
        "seasonal_force",
        (0., 0.4), jumping_stdev=0.05
    )
    # UniformPrior("vic_2021_seeding.seed_time", (530., 560.), jumping_stdev=5.)
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
