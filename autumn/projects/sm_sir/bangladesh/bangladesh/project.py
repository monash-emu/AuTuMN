import json
from pandas import Series
from datetime import datetime
import pandas as pd

from autumn.tools.dynamic_proportions.solve_transitions import calculate_transition_rates_from_dynamic_props
from autumn.tools.utils.utils import wrap_series_transform_for_ndarray
from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model
from autumn.models.sm_sir.constants import IMMUNITY_STRATA
from autumn.settings import Region, Models

# Load and configure model parameters
baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))
calibration_start_time = param_set.baseline.to_dict()["time"]["start"]

# Work out date truncation points
targets_start = (datetime(2021, 5, 15) - COVID_BASE_DATETIME).days
notifications_trunc_point = (datetime(2021, 12, 1) - COVID_BASE_DATETIME).days
notif_change_start_point = (datetime(2022, 1, 29) - COVID_BASE_DATETIME).days

# Get the actual targets
notifications_ts = ts_set["notifications"].loc[targets_start: notifications_trunc_point]
hospital_admissions_ts = ts_set["hospital_admissions"].loc[targets_start:]
late_hosp_admissions = ts_set["hospital_admissions"].loc[notifications_trunc_point:]
deaths_ts = ts_set["infection_deaths"].loc[targets_start:]
late_deaths = ts_set["infection_deaths"].loc[notifications_trunc_point:]

# Build transformed target series
def get_roc(series: Series) -> Series:
    return series.rolling(7).mean().pct_change(7)

# Build targets for rate-of-change (roc) variables
roc_vars = ["notifications"]
roc_targets = []

for roc_var in roc_vars:
    new_ts = get_roc(ts_set[roc_var]).loc[notif_change_start_point:]
    new_ts.name = roc_var + '_roc'
    roc_targets.append(NormalTarget(new_ts))

targets = [
    NormalTarget(notifications_ts),
    NormalTarget(hospital_admissions_ts),
    NormalTarget(deaths_ts),
    NormalTarget(late_hosp_admissions),
    NormalTarget(late_deaths)
] + roc_targets

priors = [
    UniformPrior("contact_rate", (0.02, 0.1)),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.005, 0.015)),
    UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (500., 550.)),
    UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (2.2, 3.5)),
    UniformPrior("age_stratification.cfr.multiplier", (0.01, 0.08)),
    UniformPrior("age_stratification.prop_hospital.multiplier", (0.01, 0.08)),
]


calibration = Calibration(priors=priors, targets=targets, random_process=None, metropolis_init="current_params")

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


def custom_build_model(param_set, build_options=None):
    model = build_model(param_set, build_options)
    for v in roc_vars:
        model.request_function_output(
            v + '_roc',
            wrap_series_transform_for_ndarray(get_roc),
            [v],
        )

    # Add in some code to track what is going on with the immunity strata, so that I can see what is going on
    for stratum in IMMUNITY_STRATA:
        n_immune_name = f"n_immune_{stratum}"
        prop_immune_name = f"prop_immune_{stratum}"
        model.request_output_for_compartments(
            n_immune_name,
            model._original_compartment_names,
            {"immunity": stratum},
        )
        model.request_function_output(
            prop_immune_name,
            lambda num, total: num / total,
            [n_immune_name, "total_population"],
        )

    # Requested proportions over time
    props_df = pd.DataFrame(
        data={
            "none": [1., .2, .2, .2],
            "low": [0., .8, .6, .7],
            "high": [0., .0, .2, .1]
        },
        index=[390, 420, 700, 900]
    )

    # List of transition flows
    active_flows = {
        "vaccination": ("none", "low"),
        "boosting": ("low", "high"),
        "waning": ("high", "low")
    }
    sc_functions = calculate_transition_rates_from_dynamic_props(props_df, active_flows)

    for comp in model._original_compartment_names:
        for transition, strata in active_flows.items():
            model.add_transition_flow(
                transition,
                sc_functions[transition],
                comp,
                comp,
                source_strata={"immunity": strata[0]},
                dest_strata={"immunity": strata[1]},
            )

    return model


# Create and register the project
project = Project(Region.BANGLADESH, Models.SM_SIR, custom_build_model, param_set, calibration, plots=plot_spec)
