from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path, use_tuned_proposal_sds
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.tuberculosis import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.tuberculosis.calibration_utils import get_natural_history_priors_from_cid
from autumn.projects.tuberculosis.kiribati.utils import make_sa_scenario_list

ANALYSIS = "main"
# ANALYSIS = "sa_importation"
# ANALYSIS = "sa_screening"


# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)

if ANALYSIS == "main":
    scenario_paths = [build_rel_path(f"params/scenario-{i}.yml") for i in range(1, 2)]
    scenario_params = [baseline_params.update(p) for p in scenario_paths]
else:
    all_scenario_dicts = make_sa_scenario_list(ANALYSIS)
    scenario_params = [baseline_params.update(p) for p in all_scenario_dicts]

param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration targets
ts_set = load_timeseries(build_rel_path("timeseries.json"))
targets = [
    NormalTarget(ts_set["prevalence_infectiousXlocation_starawa"], stdev=80.0),
    NormalTarget(ts_set["percentage_latentXlocation_starawa"], stdev=10.0),
    NormalTarget(ts_set["prevalence_infectiousXlocation_other"], stdev=20.0),
    NormalTarget(ts_set["notificationsXlocation_starawa"], stdev=20.),
    NormalTarget(ts_set["notificationsXlocation_other"], stdev=9.),
    NormalTarget(ts_set["population_size"], stdev=2500.0),
]

# Add uncertainty around natural history using our CID estimates
natural_history_priors = []
for param_name in ["infect_death_rate", "self_recovery_rate"]:
    for organ in ["smear_positive", "smear_negative"]:
        prior = get_natural_history_priors_from_cid(param_name, organ)
        natural_history_priors.append(prior)

priors = [
    # *get_dispersion_priors_for_gaussian_targets(targets),
    UniformPrior("start_population_size", [200, 800]),
    UniformPrior("contact_rate", [0.002, 0.01]),
    UniformPrior("progression_multiplier", [0.5, 2.0]),
    UniformPrior("time_variant_tb_screening_rate.inflection_time", [2000.0, 2020.0]),
    UniformPrior("time_variant_tb_screening_rate.shape", [0.07, 0.1]),
    UniformPrior("time_variant_tb_screening_rate.end_asymptote", [0.4, 0.55]),
    UniformPrior(
        "user_defined_stratifications.location.adjustments.detection.ebeye",
        [1.3, 2.0],
    ),
    UniformPrior("user_defined_stratifications.location.adjustments.detection.other", [0.5, 1.5]),
    UniformPrior("extra_params.rr_progression_diabetes", [2.0, 5.0]),
    UniformPrior("rr_infection_latent", [0.2, 0.5]),
    UniformPrior("rr_infection_recovered", [0.2, 1.0]),
    UniformPrior("pt_efficacy", [0.75, 0.85]),
    UniformPrior("awareness_raising.relative_screening_rate", [1.0, 1.5]),
    *natural_history_priors,
]

# Load proposal sds from yml file
#use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

calibration = Calibration(
    priors, targets, metropolis_init="current_params", metropolis_init_rel_step_size=0.1
)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

diff_output_requests = [
    ["cumulative_diseased", "ABSOLUTE"],
    ["cumulative_deaths", "ABSOLUTE"],
    ["cumulative_pt", "ABSOLUTE"],
    ["cumulative_pt_sae", "ABSOLUTE"],
]

project = Project(
    Region.KIRIBATI,
    Models.TB,
    build_model,
    param_set,
    calibration,
    plots=plot_spec,
    diff_output_requests=diff_output_requests,
)
