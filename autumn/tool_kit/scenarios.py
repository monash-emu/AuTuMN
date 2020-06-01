"""
Utilities for running multiple model scenarios
"""
import numpy
from copy import deepcopy
from typing import Callable

from summer.model import StratifiedModel

from autumn.tool_kit import schema_builder as sb
from autumn.tool_kit.timer import Timer

from ..constants import IntegrationType

from .utils import merge_dicts

validate_params = sb.build_validator(
    default=dict, scenario_start_time=float, scenarios=dict
)

ModelBuilderType = Callable[[dict], StratifiedModel]


class Scenario:
    """
    A particular run of a simulation using a common model and unique parameters.
    """

    def __init__(
        self, model_builder: ModelBuilderType, idx: str, params: dict, chain_idx=0
    ):
        _params = deepcopy(params)
        validate_params(_params)
        self.model_builder = model_builder
        self.idx = idx
        self.chain_idx = chain_idx
        self.name = "baseline" if idx == 0 else f"scenario-{idx}"
        self.params = _params
        self.generated_outputs = None

    @classmethod
    def load_from_db(
        self, idx: int, chain_idx: int, model: StratifiedModel, params=None
    ):
        """
        Construct a Scenario from a model that's been loaded from an output database.
        """
        empty_params = {"default": {}, "scenario_start_time": 0, "scenarios": {}}
        params = params or empty_params
        scenario = Scenario(None, idx, params, chain_idx=chain_idx)
        scenario.model = model
        return scenario

    def run(self, base_model=None, update_func=None):
        """
        Run the scenario model simulation.
        If a base model is provided, then run the scenario from the scenario start time.
        If a parameter update function is provided, it will be used to update params before the model is run.
        """
        with Timer(f"Running scenario: {self.name}"):
            params = None
            if not base_model:
                # This model is the baseline model
                assert self.is_baseline, "Can only run base model if Scenario idx is 0"
                params = self.params["default"]
                if update_func:
                    # Apply extra parameter updates
                    params = update_func(params)

                self.model = self.model_builder(params)
            else:
                # This is a scenario model, based off the baseline model
                assert (
                    not self.is_baseline
                ), "Can only run scenario model if Scenario idx is > 0"

                # Construct scenario params by merging scenario-specific params into default params
                default_params = self.params["default"]
                scenario_params = self.params["scenarios"][self.idx]
                params = merge_dicts(scenario_params, default_params)

                if update_func:
                    # Apply extra parameter updates
                    params = update_func(params)

                # Override start time.
                params = {**params, "start_time": self.params["scenario_start_time"]}

                base_times = base_model.times
                base_outputs = base_model.outputs

                # Find the time step from which we will start the scenario
                start_index = get_scenario_start_index(base_times, params["start_time"])
                start_time = base_times[start_index]
                init_compartments = base_outputs[start_index, :]

                # Create the new scenario model using the scenario-specific params,
                # ensuring the initial conditions are the same for the given start time.
                self.model = self.model_builder(params)
                self.model.compartment_values = init_compartments

            self.model.run_model(IntegrationType.SOLVE_IVP)

    @property
    def is_baseline(self):
        """Return True if this is a baseline model."""
        return self.idx == 0

    @property
    def has_run(self):
        """Return True if the model has been run."""
        return self.model and self.model.outputs is not None


def get_scenario_start_index(base_times, scenario_start_time):
    """
    Returns the index of the closest time step that is at, or before the scenario start time.
    """
    indices_after_start_index = [
        idx for idx, time in enumerate(base_times) if time > scenario_start_time
    ]
    if not indices_after_start_index:
        raise ValueError(
            f"Scenario start time {scenario_start_time} is set after the baseline time range."
        )

    index_after_start_index = min(indices_after_start_index)
    start_index = max([0, index_after_start_index - 1])
    return start_index


def get_model_times_from_inputs(start_time, end_time, time_step):
    """
    Find the time steps for model integration from the submitted requests, ensuring the time points are evenly spaced.
    """
    n_times = int(round((end_time - start_time) / time_step)) + 1
    return numpy.linspace(start_time, end_time, n_times).tolist()
