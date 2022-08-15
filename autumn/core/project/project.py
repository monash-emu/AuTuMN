import glob
import inspect
import json
import logging
import os
import re
from datetime import datetime
from importlib import import_module
from importlib import reload as reload_module
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from autumn.core.db.database import FeatherDatabase
from autumn.core.db.store import (
    Table,
    build_derived_outputs_table,
    build_outputs_table,
    save_model_outputs,
)
from autumn.core.project.params import read_yaml_file
from autumn.core.registry import _PROJECTS
from autumn.core.utils.git import get_git_branch, get_git_hash
from autumn.core.utils.timer import Timer
from autumn.settings import BASE_PATH, DOCS_PATH, MODELS_PATH, OUTPUT_DATA_PATH
from summer.derived_outputs import DerivedOutputRequest
from summer.model import CompartmentalModel

from .params import ParameterSet, Params

logger = logging.getLogger(__name__)

ModelBuilder = Callable[[dict, dict], CompartmentalModel]


class Project:
    """
    A project that combined a model, a set of parameters and a calibration approach.
    """

    def __init__(
        self,
        region_name: str,
        model_name: str,
        build_model: ModelBuilder,
        param_set: ParameterSet,
        calibration,  # A Calibration instance
        plots: dict = None,  # Previously, targets JSON.
        diff_output_requests: List[Tuple[str, str]] = None,
        ts_set: dict = None,
    ):
        self.region_name = region_name
        self.model_name = model_name
        self.build_model = build_model
        self.param_set = param_set
        self.plots = plots or {}
        self.calibration = calibration
        self.diff_output_requests = diff_output_requests or []
        self.ts_set = ts_set or None

    def calibrate(self, max_seconds: float, chain_idx: int, num_chains: int):
        """
        Calibrate the model using the baseline parameters.
        """
        with Timer(f"Running calibration for {self.model_name} {self.region_name}."):
            self.calibration.run(self, max_seconds, chain_idx, num_chains)

    def run_baseline_model(
        self,
        params: Params,
        derived_outputs_whitelist: Optional[List[str]] = None,
        build_options: Optional[dict] = None,
    ) -> CompartmentalModel:
        """
        Run the project's baseline model with the given parameters.
        Returns the completed baseline model.
        """
        params_dict = params.to_dict()
        model = self.build_model(params_dict, build_options)
        if type(model) is CompartmentalModel and derived_outputs_whitelist:
            # Only calculate required derived outputs.
            model.set_derived_outputs_whitelist(derived_outputs_whitelist)

        self._run_model(model)
        return model

    def run_scenario_models(
        self,
        baseline_model: CompartmentalModel,
        scenario_params: List[Params],
        start_time: Optional[float] = None,
        start_times: Optional[List[float]] = None,
        build_options: Optional[List[dict]] = None,
    ) -> List[CompartmentalModel]:
        """
        Runs all the project's scenarios with the given parameters.
        Returns the completed scenario models.
        """
        # Figure out what start times to use for each scenario.
        if start_times is None and start_time is not None:
            # Use the same start time for each scenario.
            start_times = [start_time] * len(scenario_params)
        elif start_times is None:
            # No start times specified - use whatever the model defaults are.
            start_times = [None] * len(scenario_params)

        if build_options is None:
            build_options = [None] * len(scenario_params)

        models = []
        assert baseline_model.outputs is not None, "Baseline mode has not been run yet."
        for start_time, params, build_opt in zip(start_times, scenario_params, build_options):

            params_dict = params.to_dict()
            model = self.build_model(params_dict, build_opt)

            if start_time is not None:
                # Find the initial conditions for the given start time
                start_idx = get_scenario_start_index(baseline_model.times, start_time)
                init_compartments = baseline_model.outputs[start_idx, :]
                # Use initial conditions at the given start time.
                if type(model) is CompartmentalModel:
                    model.initial_population = init_compartments
                else:
                    model.compartment_values = init_compartments

            self._run_model(model)
            models.append(model)

        return models

    def _run_model(self, model: CompartmentalModel):
        """
        Run the model.
        """
        model.run(max_step=1)

    def write_params_to_tex(self, main_table_params_list, project_path, output_dir_path=None):
        """
        Write the main parameter table as a tex file. Also write the table of calibrated parameters in a separate tex file.
        :param main_table_params_list: ordered list of parameters to be included in the main table
        :param project_path: path of the project's directory
        :param output_dir_path: path of the directory where to dump the output tex files.
               Default is "docs/papers/<model_name>/projects/<region_name>".
        """
        # Load parameters' descriptions (base model)
        base_params_descriptions_path = os.path.join(
            MODELS_PATH, self.model_name, "params_descriptions.json"
        )
        with open(base_params_descriptions_path, mode="r") as f:
            params_descriptions = json.load(f)

        # Load parameters' descriptions (project-specific)
        updated_descriptions_path = os.path.join(project_path, "params_descriptions.json")
        if os.path.isfile(updated_descriptions_path):
            with open(updated_descriptions_path, mode="r") as f:
                updated_params_descriptions = json.load(f)
            params_descriptions.update(updated_params_descriptions)

        # work out output dir path
        if output_dir_path is None:
            output_dir_path = os.path.join(
                DOCS_PATH, "papers", self.model_name, "projects", self.region_name
            )

        # Get list of priors
        all_calibration_params_names = (
            self.calibration.iterative_sampling_param_names
            + self.calibration.independent_sampling_param_names
        )
        all_priors = (
            self.calibration.iterative_sampling_priors
            + self.calibration.independent_sampling_priors
        )

        # Write main parameter table to tex file
        write_main_param_table(
            self,
            main_table_params_list,
            params_descriptions,
            all_calibration_params_names,
            all_priors,
            output_dir_path,
        )

        # Write calibrated parameter table to tex file
        write_priors_table(params_descriptions, all_priors, output_dir_path)

    def get_path(self) -> Path:
        """
        Return a pathlib.Path to the current project directory
        """
        return Path(BASE_PATH) / "/".join(self._get_path().split(".")[:-1])

    def __repr__(self):
        return f"Project<{self.model_name}, {self.region_name}>"

    def _get_path(self):
        return _PROJECTS[self.model_name][self.region_name]


LOADED_PROJECTS = set()


def get_project(model_name: str, project_name: str, reload=False) -> Project:
    """
    Returns a registered project
    """
    assert model_name in _PROJECTS, f"Model {model_name} not registered as a project."
    msg = f"Project {project_name} not registered as a project using model {model_name}."
    assert project_name in _PROJECTS[model_name], msg
    import_path = _PROJECTS[model_name][project_name]

    project_module = import_module(import_path)
    if import_path in LOADED_PROJECTS and reload:
        reload_module(project_module)

    try:
        project = project_module.project
    except (AttributeError, AssertionError):
        msg = f"Cannot find a Project instance named 'project' in {import_path}"
        raise ImportError(msg)

    LOADED_PROJECTS.add(import_path)
    return project


def build_rel_path(path):
    """
    Returns the absolute path of a file, relative to the file
    where this function is being called from.
    """
    # Find the file path of the calling file.
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    file_path = module.__file__
    # Find directory that the calling file lives in
    file_dirpath = os.path.dirname(file_path)
    # Append the provied path to the directory path, ensure absolute.
    return os.path.abspath(os.path.join(file_dirpath, path))


def get_scenario_start_index(base_times: List[float], start_time: float):
    """
    Returns the index of the closest time step that is at, or before the scenario start time.
    """
    msg = f"Scenario start time {start_time} is before baseline has started"
    assert base_times[0] <= start_time, msg
    indices_after_start_index = [idx for idx, time in enumerate(base_times) if time > start_time]
    if not indices_after_start_index:
        raise ValueError(f"Scenario start time {start_time} is set after the baseline time range")

    index_after_start_index = min(indices_after_start_index)
    start_index = max([0, index_after_start_index - 1])
    return start_index


def run_project_locally(project: Project, run_scenarios=True):
    """
    Runs the model for the given region, storing the outputs on disk.
    This is for when you want to just run the model on your laptop to see the outputs.
    """
    logger.info(f"Running {project.model_name} {project.region_name}...")

    # Ensure project folder exists.
    project_dir = os.path.join(OUTPUT_DATA_PATH, "run", project.model_name, project.region_name)
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    output_dir = os.path.join(project_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Determine where to save model outputs
    output_db_path = os.path.join(output_dir, "outputs")

    # Save model parameters to output dir.
    param_path = os.path.join(output_dir, "params.yml")
    with open(param_path, "w") as f:
        yaml.dump(project.param_set.dump_to_dict(), f)

    # Save model run metadata to output dir.
    meta_path = os.path.join(output_dir, "meta.yml")
    metadata = {
        "model_name": project.model_name,
        "region_name": project.region_name,
        "start_time": timestamp,
        "git_branch": get_git_branch(),
        "git_commit": get_git_hash(),
    }
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)

    with Timer("Running baseline model"):
        baseline_model = project.run_baseline_model(project.param_set.baseline)

    if run_scenarios:
        num_scenarios = len(project.param_set.scenarios)
        with Timer(f"Running {num_scenarios} model scenarios"):
            start_times = [
                sc_params.to_dict()["time"]["start"] for sc_params in project.param_set.scenarios
            ]
            sc_models = project.run_scenario_models(
                baseline_model, project.param_set.scenarios, start_times=start_times
            )
    else:
        sc_models = []

    with Timer("Saving model outputs to the database"):
        outputs_db = FeatherDatabase(output_db_path)
        all_outputs = post_process_scenario_outputs([baseline_model, *sc_models], project)
        save_model_outputs(outputs_db, **all_outputs)


class DiffOutput:
    RELATIVE = "RELATIVE"
    ABSOLUTE = "ABSOLUTE"
    REQUEST_TYPES = [RELATIVE, ABSOLUTE]


def get_all_available_scenario_paths(scenario_dir_path):
    """
    Automatically lists the paths of all the yml files starting with 'scenario-' available in a given directory.
    :param scenario_dir_path: path to the directory
    :return: a list of paths
    """
    glob_str = os.path.join(scenario_dir_path, "scenario-*.yml")
    scenario_file_list = glob.glob(glob_str)

    # Sort by integer rather than string (so that 'scenario-2' comes before 'scenario-10')
    file_list_sorted = sorted(
        scenario_file_list, key=lambda x: int(re.match(".*scenario-([0-9]*)", x).group(1))
    )

    return file_list_sorted


def use_tuned_proposal_sds(priors, proposal_sds_path):
    if os.path.isfile(proposal_sds_path):
        proposal_sds = read_yaml_file(proposal_sds_path)
        for prior in priors:
            if prior.name in proposal_sds:
                if proposal_sds[prior.name] is not None:
                    prior.jumping_stdev = proposal_sds[prior.name]


def post_process_scenario_outputs(
    models: List[CompartmentalModel], project: Project, run_id: int = 0, chain_id: int = None
) -> Dict[str, pd.DataFrame]:
    """
    Do any required postprocessing of scenario outputs,
    particularly those that require comparison to baseline.

    Args:
        models (List[CompartmentalModel]): List of (run) models, as returned from run_baseline_model / run_scenario_models
        run_id (int, optional): Required for multiple (usually remote) runs, not required for single local runs
        chain_id (int, optional): Used by MCMC runs, not required for single runs

    Returns:
        dict: Dict whose keys are our expected table names, and values are the (processed) DataFrames
    """
    # Apply various post processing fixes
    calculate_differential_outputs(models, project.diff_output_requests)
    fix_cumulative_output_times(models)

    # Build outputs for storage in a database.
    outputs_df = build_outputs_table(models, run_id=run_id, chain_id=chain_id)
    derived_outputs_df = build_derived_outputs_table(models, run_id=run_id, chain_id=chain_id)
    return {
        Table.OUTPUTS: outputs_df,
        Table.DERIVED: derived_outputs_df,
    }


def fix_cumulative_output_times(models: List[CompartmentalModel]):
    """
    Fix a bug with summer's cumulative outputs
    FIXME: Accessing private member of model class; prefer not to modify summer code just for this
    """
    baseline_model = models[0]
    if type(baseline_model) is CompartmentalModel:
        cum_out_keys = [
            k
            for k, req in baseline_model._derived_output_requests.items()
            if req["request_type"] == DerivedOutputRequest.CUMULATIVE
            and req["save_results"] == True
        ]
    else:
        cum_out_keys = [k for k in baseline_model.derived_outputs.keys() if k.startswith("cum")]

    for scenario_model in models[1:]:
        baseline_start_index = get_scenario_start_index(
            baseline_model.times, scenario_model.times[0]
        )
        # Adjust cumulative outputs to start at baseline value rather than 0
        for output_key in cum_out_keys:
            baseline_offset = baseline_model.derived_outputs[output_key][baseline_start_index]
            scenario_model.derived_outputs[output_key] += baseline_offset


def calculate_differential_outputs(
    models: List[CompartmentalModel], diff_output_requests: List[Tuple[str, str]]
):
    """
    Calculate the difference in derived outputs between scenarios.
    For example, how many lives saved between Scenario 1 and baseline.
    Updates the model's derived outputs in-place.

    Models list assumed to be in order [baseline, sc1, sc2, ..., scN]

    diff_output_requests has form
    {
        "<output name>": DiffOutput.<request type>,
        "<output name>": DiffOutput.<request type>,
        "<output name>": DiffOutput.<request type>,
        "<output name>": DiffOutput.<request type>,
    }
    """
    baseline_model = models[0]
    for diff_output_request in diff_output_requests:
        output_name, request = diff_output_request
        assert request in DiffOutput.REQUEST_TYPES, f"Request {request} unknown."
        # Evaluate each request.
        for model in models:
            # Evaluate each model (including baseline) compared to baseline.
            msg = f"Derived output {output_name} missing in baseline model."
            assert output_name in baseline_model.derived_outputs, msg
            msg = f"Derived output {output_name} missing in scenario model."
            assert output_name in model.derived_outputs, msg
            baseline_output = baseline_model.derived_outputs[output_name]
            sc_output = model.derived_outputs[output_name]
            idx_shift = baseline_output.size - sc_output.size
            output_arr = np.zeros(sc_output.shape)
            calc = OUTPUT_CALCS[request]
            new_output_name = calc(output_name, output_arr, baseline_output, sc_output, idx_shift)
            model.derived_outputs[new_output_name] = output_arr


def calc_relative_diff_output(output_name, output_arr, baseline_output, sc_output, idx_shift):
    for idx in range(len(sc_output)):
        sc_val = sc_output[idx]
        baseline_val = baseline_output[idx + idx_shift]
        if baseline_val == 0:
            # Zero for undefined diffs
            output_arr[idx] = 0
        else:
            output_arr[idx] = (sc_val - baseline_val) / baseline_val * 100.0

    new_output_name = f"rel_diff_{output_name}"
    return new_output_name


def calc_absolute_diff_output(output_name, output_arr, baseline_output, sc_output, idx_shift):
    for idx in range(len(sc_output)):
        sc_val = sc_output[idx]
        baseline_val = baseline_output[idx + idx_shift]
        output_arr[idx] = sc_val - baseline_val

    new_output_name = f"abs_diff_{output_name}"
    return new_output_name


OUTPUT_CALCS = {
    DiffOutput.RELATIVE: calc_relative_diff_output,
    DiffOutput.ABSOLUTE: calc_absolute_diff_output,
}
