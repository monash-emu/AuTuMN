import json
import numpy as np

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, get_all_available_scenario_paths
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.tools.calibration.targets import PoissonTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters
default_path = build_rel_path("params/default.yml")
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = (base_params.update(default_path).update(mle_path, calibration_format=True))

scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings
ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))
notifications = ts_set.get("notifications").truncate_times(551, 707)
infection_deaths = ts_set.get("infection_deaths").truncate_times(551, 706)
hosp_admissions = ts_set.get("hospital_admissions").truncate_times(551, 707)
icu_admissions = ts_set.get("icu_admissions").truncate_times(551, 707)
targets = [
    PoissonTarget(notifications),
    PoissonTarget(infection_deaths),
    PoissonTarget(hosp_admissions),
    PoissonTarget(icu_admissions)
]
incubation_period_string = "sojourn.compartment_periods_calculated.exposed.total_period"
hosp_multiplier_string = "clinical_stratification.props.hospital.multiplier"
priors = [
    UniformPrior("contact_rate", (0.05, 0.1), jumping_stdev=0.01),
    TruncNormalPrior(incubation_period_string, mean=3.5, stdev=1.0, trunc_range=(2.0, np.inf)),
    UniformPrior("vaccination.vacc_full_effect_duration", (60., 180.), jumping_stdev=15.),
    UniformPrior("vaccination.vacc_part_effect_duration", (180., 360.), jumping_stdev=30.),
    UniformPrior("mobility.microdistancing.face_coverings_adjuster.parameters.effect", (0.05, 0.3), jumping_stdev=0.04),
    UniformPrior("infectious_seed", (2., 100.), jumping_stdev=30.),
    TruncNormalPrior(hosp_multiplier_string, mean=1.0, stdev=0.5, trunc_range=(0.8, np.inf)),
]

calibration = Calibration(priors=priors, targets=targets)

plot_spec_filepath = build_rel_path("targets.secret.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(Region.VICTORIA, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec)
