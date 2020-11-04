"""
Utilities for running multiple model scenarios
"""
import numpy as np
from copy import deepcopy
from typing import Callable, List

from summer.model import StratifiedModel

from autumn.tool_kit import schema_builder as sb
from autumn.tool_kit.timer import Timer
from autumn.tool_kit.params import update_params

from ..constants import IntegrationType

validate_params = sb.build_validator(default=dict, scenarios=dict)

ModelBuilderType = Callable[[dict], StratifiedModel]


class Scenario:
    """
    A particular run of a simulation using a common model and unique parameters.
    """

    def __init__(self, model_builder: ModelBuilderType, idx: str, params: dict):
        _params = deepcopy(params)
        validate_params(_params)
        self.model_builder = model_builder
        self.idx = idx
        self.name = "baseline" if idx == 0 else f"scenario-{idx}"
        self.params = _params
        self.generated_outputs = None

    @classmethod
    def load_from_db(self, idx: int, model: StratifiedModel, params=None):
        """
        Construct a Scenario from a model that's been loaded from an output database.
        """
        empty_params = {"default": {}, "scenarios": {}}
        params = params or empty_params
        scenario = Scenario(None, idx, params)
        scenario.model = model
        return scenario

    def run(self, base_model=None, update_func=None, _hack_in_scenario_params: dict = None):
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
                assert not self.is_baseline, "Can only run scenario model if Scenario idx is > 0"

                # Construct scenario params by merging scenario-specific params into default params
                params = self.params["scenarios"][self.idx]
                start_time = params["time"]["start"]
                if update_func:
                    # Apply extra parameter updates
                    params = update_func(params)

                if _hack_in_scenario_params:
                    # Hack in scenario params for mixing optimization project.
                    # TODO: Refactor code so that scenario params are applied *after* calibration update.
                    params = update_params(params, _hack_in_scenario_params)

                # Ensure start time cannot be overwritten for a scenario
                params["time"]["start"] = start_time

                base_times = base_model.times
                base_outputs = base_model.outputs

                # Find the time step from which we will start the scenario
                start_index = get_scenario_start_index(base_times, params["time"]["start"])
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


def get_scenario_start_index(base_times, start_time):
    """
    Returns the index of the closest time step that is at, or before the scenario start time.
    """
    indices_after_start_index = [idx for idx, time in enumerate(base_times) if time > start_time]
    if not indices_after_start_index:
        raise ValueError(f"Scenario start time {start_time} is set after the baseline time range.")

    index_after_start_index = min(indices_after_start_index)
    start_index = max([0, index_after_start_index - 1])
    return start_index


def get_model_times_from_inputs(start_time, end_time, time_step, critical_ranges=[]):
    """
    Find the time steps for model integration from the submitted requests, ensuring the time points are evenly spaced.
    Use a refined time-step within critical ranges
    """
    times = []
    interval_start = start_time
    for critical_range in critical_ranges:
        # add regularly-spaced points up until the start of the critical range
        interval_end = critical_range[0]
        if interval_end > interval_start:
            times += list(np.arange(interval_start, interval_end, time_step))
        # add points over the critical range with smaller time step
        interval_start = interval_end
        interval_end = critical_range[1]
        if interval_end > interval_start:
            times += list(np.arange(interval_start, interval_end, time_step / 10.0))
        interval_start = interval_end

    if end_time > interval_start:
        times += list(np.arange(interval_start, end_time, time_step))
    times.append(end_time)

    # clean up time values ending .9999999999
    times = [round(t, 5) for t in times]

    return np.array(times)


def calculate_differential_outputs(models: List[StratifiedModel], targets: dict):
    """
    :param models: list of fully run models.
    :param targets: dictionary containing targets.
    :return: list of models with additional derived_outputs
    """
    baseline_derived_outputs = models[0].derived_outputs

    baseline_derived_output_keys = list(baseline_derived_outputs.keys())
    for derived_ouptut in baseline_derived_output_keys:
        target = targets.get(derived_ouptut)
        if not target:
            continue

        diff_type = target.get("differentiate")
        if not diff_type:
            continue

        baseline_output = baseline_derived_outputs[derived_ouptut]
        for model in models:
            sc_output = model.derived_outputs[derived_ouptut]
            idx_shift = len(baseline_output) - len(sc_output)
            output_arr = np.zeros(sc_output.shape)
            if diff_type == "relative":
                new_output_name = f"rel_diff_{derived_ouptut}"
                for idx in range(len(sc_output)):
                    sc_val = sc_output[idx]
                    baseline_val = baseline_output[idx + idx_shift]
                    if baseline_val == 0:
                        # Zero for undefined diffs
                        output_arr[idx] = 0
                    else:
                        output_arr[idx] = (sc_val - baseline_val) / baseline_val * 100.0

            elif diff_type == "absolute":
                new_output_name = f"abs_diff_{derived_ouptut}"
                for idx in range(len(sc_output)):
                    sc_val = sc_output[idx]
                    baseline_val = baseline_output[idx + idx_shift]
                    output_arr[idx] = sc_val - baseline_val
            else:
                raise ValueError("Differential outputs must be 'relative' or 'absolute'")

            model.derived_outputs[new_output_name] = output_arr

    return models
