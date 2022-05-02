import numpy as np

from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.targets import (
    TruncNormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.covid_19.mixing_optimisation.priors import PRIORS
from autumn.projects.covid_19.mixing_optimisation.constants import (
    CALIBRATION_END,
    CALIBRATION_START,
)

# Load and configure model parameters.
baseline_params = base_params.update(
    build_rel_path("../eur_optimisation_params.yml"), validate=False
).update(build_rel_path("params/default.yml"))
param_set = ParameterSet(baseline=baseline_params)

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))

calib_times = (CALIBRATION_START, CALIBRATION_END)
targets = [
    TruncNormalTarget(
        data=ts_set["notifications"].loc[CALIBRATION_START:CALIBRATION_END],
        trunc_range=[0, np.inf],
    ),
    TruncNormalTarget(
        data=ts_set["infection_deaths"].loc[CALIBRATION_START:CALIBRATION_END],
        trunc_range=[0, np.inf],
    ),
    TruncNormalTarget(
        data=ts_set["hospital_admissions"].loc[CALIBRATION_START:CALIBRATION_END],
        trunc_range=[0, np.inf],
    ),
    TruncNormalTarget(
        data=ts_set["prop_ever_infected"].loc[CALIBRATION_START:CALIBRATION_END],
        trunc_range=[0, 1],
        stdev=0.04,
    ),
]
priors = [*PRIORS, *get_dispersion_priors_for_gaussian_targets(targets)]
calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(
    Region.SPAIN, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
