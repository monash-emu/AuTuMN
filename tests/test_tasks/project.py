from summer.model import CompartmentalModel

from autumn.calibration.priors import UniformPrior
from autumn.calibration.import NormalTarget

from autumn.tools.project import Project, ParameterSet, load_timeseries, Params
from autumn.runners import Calibration


def get_test_project():
    baseline_params_data = {"time": {"start": 0}, "birth_rate": 0.1, "recovery_rate": 0.1}
    scenario_params_data = {"time": {"start": 2}, "birth_rate": 0.2}
    baseline_params = Params(baseline_params_data)
    scenario_params = baseline_params.update(scenario_params_data)
    param_set = ParameterSet(baseline=baseline_params, scenarios=[scenario_params])

    ts_data = {
        "recovery": {
            "times": [0, 1, 2, 3, 4, 5],
            "values": [275, 240, 180, 140, 110, 80],
        }
    }
    ts_set = TimeSeriesSet(ts_data)
    priors = [UniformPrior("recovery_rate", [0.05, 0.5])]
    targets = [NormalTarget(ts_set["recovery"])]
    calibration = Calibration(priors=priors, targets=targets, seed=0)
    project = Project("test_region", "test_model", build_test_model, param_set, calibration)
    return project


def build_test_model(params: dict, build_options: dict = None):
    model = CompartmentalModel(
        times=[params["time"]["start"], 5],
        compartments=["S", "I", "R"],
        infectious_compartments=["I"],
    )
    model.set_initial_population(distribution={"S": 1000, "I": 1000})
    model.add_crude_birth_flow("birth", params["birth_rate"], "S")
    model.add_transition_flow("recovery", params["recovery_rate"], "I", "R")
    model.request_output_for_flow("recovery", "recovery")
    return model
