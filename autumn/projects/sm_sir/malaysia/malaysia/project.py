import json

from autumn.core.project import Project, ParameterSet, load_timeseries, build_rel_path,\
    get_all_available_scenario_paths
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.sm_sir.malaysia.malaysia.scenario_builder import get_all_scenario_dicts

# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)
scenario_dir_path = build_rel_path("params/")
#scenario_paths = get_all_available_scenario_paths(scenario_dir_path)
all_scenario_dicts = get_all_scenario_dicts("MYS")
scenario_params = [baseline_params.update(sc_dict) for sc_dict in all_scenario_dicts]
#scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings
calibration_start_time = 426 #param_set.baseline.to_dict()["time"]["start"]
ts_set = load_timeseries(build_rel_path("timeseries.json"))
notifications_end_time = 762
notifications_ts = ts_set["notifications"].loc[calibration_start_time:]
icu_occupancy_ts = ts_set["icu_occupancy"].loc[calibration_start_time:]
hospital_occupancy_ts = ts_set["hospital_occupancy"].loc[calibration_start_time:]
infection_deaths_ts = ts_set["infection_deaths"].loc[calibration_start_time:]
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(icu_occupancy_ts),
    NormalTarget(hospital_occupancy_ts),
    NormalTarget(infection_deaths_ts)
]

priors = [
    UniformPrior("contact_rate", (0.01, 0.1)),
    UniformPrior("detect_prop", (0.07, 0.3)),
    UniformPrior("sojourns.latent.total_time", (5, 20)),
    UniformPrior("voc_emergence.delta.contact_rate_multiplier", (1.5, 3)),
    UniformPrior("voc_emergence.delta.new_voc_seed.start_time", (400, 500)),
    UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (2, 3.5)),
    UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (520, 675)),
    UniformPrior("voc_emergence.omicron.relative_latency", (0.45, 0.75)),
    UniformPrior("voc_emergence.omicron.relative_active_period", (0.5, 1.5)),
    UniformPrior("sojourns.recovered", (150, 300)),
    UniformPrior("immunity_stratification.infection_risk_reduction.low", (0.5, 0.64)),
    UniformPrior("immunity_stratification.infection_risk_reduction.high", (0.65, 0.8)),

]
diff_output_requests = [
    ["notificationsXagegroup_0", "ABSOLUTE"],
    ["notificationsXagegroup_5", "ABSOLUTE"],
    ["notificationsXagegroup_10", "ABSOLUTE"],
    ["notificationsXagegroup_15", "ABSOLUTE"],
    ["notificationsXagegroup_20", "ABSOLUTE"],
    ["notificationsXagegroup_25", "ABSOLUTE"],
    ["notificationsXagegroup_30", "ABSOLUTE"],
    ["notificationsXagegroup_35", "ABSOLUTE"],
    ["notificationsXagegroup_40", "ABSOLUTE"],
    ["notificationsXagegroup_45", "ABSOLUTE"],
    ["notificationsXagegroup_50", "ABSOLUTE"],
    ["notificationsXagegroup_55", "ABSOLUTE"],
    ["notificationsXagegroup_60", "ABSOLUTE"],
    ["notificationsXagegroup_65", "ABSOLUTE"],
    ["notificationsXagegroup_70", "ABSOLUTE"],
    ["notificationsXagegroup_75", "ABSOLUTE"],
    ["infection_deathsXagegroup_0", "ABSOLUTE"],
    ["infection_deathsXagegroup_5", "ABSOLUTE"],
    ["infection_deathsXagegroup_10", "ABSOLUTE"],
    ["infection_deathsXagegroup_15", "ABSOLUTE"],
    ["infection_deathsXagegroup_20", "ABSOLUTE"],
    ["infection_deathsXagegroup_25", "ABSOLUTE"],
    ["infection_deathsXagegroup_30", "ABSOLUTE"],
    ["infection_deathsXagegroup_35", "ABSOLUTE"],
    ["infection_deathsXagegroup_40", "ABSOLUTE"],
    ["infection_deathsXagegroup_45", "ABSOLUTE"],
    ["infection_deathsXagegroup_50", "ABSOLUTE"],
    ["infection_deathsXagegroup_55", "ABSOLUTE"],
    ["infection_deathsXagegroup_60", "ABSOLUTE"],
    ["infection_deathsXagegroup_65", "ABSOLUTE"],
    ["infection_deathsXagegroup_70", "ABSOLUTE"],
    ["infection_deathsXagegroup_75", "ABSOLUTE"],
    ["icu_occupancyXagegroup_0", "ABSOLUTE"],
    ["icu_occupancyXagegroup_5", "ABSOLUTE"],
    ["icu_occupancyXagegroup_10", "ABSOLUTE"],
    ["icu_occupancyXagegroup_15", "ABSOLUTE"],
    ["icu_occupancyXagegroup_20", "ABSOLUTE"],
    ["icu_occupancyXagegroup_25", "ABSOLUTE"],
    ["icu_occupancyXagegroup_30", "ABSOLUTE"],
    ["icu_occupancyXagegroup_35", "ABSOLUTE"],
    ["icu_occupancyXagegroup_40", "ABSOLUTE"],
    ["icu_occupancyXagegroup_45", "ABSOLUTE"],
    ["icu_occupancyXagegroup_50", "ABSOLUTE"],
    ["icu_occupancyXagegroup_55", "ABSOLUTE"],
    ["icu_occupancyXagegroup_60", "ABSOLUTE"],
    ["icu_occupancyXagegroup_65", "ABSOLUTE"],
    ["icu_occupancyXagegroup_70", "ABSOLUTE"],
    ["icu_occupancyXagegroup_75", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_0", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_5", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_10", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_15", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_20", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_25", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_30", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_35", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_40", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_45", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_50", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_55", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_60", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_65", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_70", "ABSOLUTE"],
    ["hospital_occupancyXagegroup_75", "ABSOLUTE"],
    ["icu_admissionsXagegroup_0", "ABSOLUTE"],
    ["icu_admissionsXagegroup_5", "ABSOLUTE"],
    ["icu_admissionsXagegroup_10", "ABSOLUTE"],
    ["icu_admissionsXagegroup_15", "ABSOLUTE"],
    ["icu_admissionsXagegroup_20", "ABSOLUTE"],
    ["icu_admissionsXagegroup_25", "ABSOLUTE"],
    ["icu_admissionsXagegroup_30", "ABSOLUTE"],
    ["icu_admissionsXagegroup_35", "ABSOLUTE"],
    ["icu_admissionsXagegroup_40", "ABSOLUTE"],
    ["icu_admissionsXagegroup_45", "ABSOLUTE"],
    ["icu_admissionsXagegroup_50", "ABSOLUTE"],
    ["icu_admissionsXagegroup_55", "ABSOLUTE"],
    ["icu_admissionsXagegroup_60", "ABSOLUTE"],
    ["icu_admissionsXagegroup_65", "ABSOLUTE"],
    ["icu_admissionsXagegroup_70", "ABSOLUTE"],
    ["icu_admissionsXagegroup_75", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_0", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_5", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_10", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_15", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_20", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_25", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_30", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_35", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_40", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_45", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_50", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_55", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_60", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_65", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_70", "ABSOLUTE"],
    ["hospital_admissionsXagegroup_75", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_0", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_5", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_10", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_15", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_20", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_25", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_30", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_35", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_40", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_45", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_50", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_55", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_60", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_65", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_70", "ABSOLUTE"],
    ["non_hosp_notificationsXagegroup_75", "ABSOLUTE"],
]

calibration = Calibration(
    priors=priors, targets=targets, random_process=None, metropolis_init="current_params"
)

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

# Create and register the project
project = Project(
    Region.MALAYSIA, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec,
    diff_output_requests=diff_output_requests
)
