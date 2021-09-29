import numpy as np
import numpy as np

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.tools.calibration.targets import NormalTarget, PoissonTarget
from autumn.models.example import base_params, build_model
from autumn.settings import Region, Models


# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
mle_path = build_rel_path("params/mle-params.yml")
scenario_paths = [build_rel_path("params/scenario-1.yml")]
baseline_params = base_params.update(default_path)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=[scenario_params])

ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))
target_start_time = 600

# Load and configure calibration settings.
ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))
priors = [
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
]
targets = [
    PoissonTarget(
        ts_set.get("notifications").round_values().truncate_start_time(target_start_time)
    ),
    PoissonTarget(ts_set.get("hospital_admissions").truncate_start_time(target_start_time)),
    PoissonTarget(ts_set.get("icu_admissions").truncate_start_time(target_start_time)),
]
# Load proposal sds from yml file
# use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

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

project = Project(Region.NORTH_METRO, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec)
