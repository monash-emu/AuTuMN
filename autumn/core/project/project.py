import glob
import inspect
import json
import logging
import os
import re
from datetime import datetime, timedelta
from importlib import import_module
from importlib import reload as reload_module
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from sys import stdout

from computegraph.utils import expand_nested_dict

import numpy as np
import pandas as pd
import yaml
from autumn.core.db.database import FeatherDatabase

from summer2.model import CompartmentalModel as CompartmentalModel2

from autumn.core.db.store import (
    Table,
    build_derived_outputs_table,
    build_outputs_table,
    save_model_outputs,
)
from autumn.core.project.params import read_yaml_file
from autumn.core.registry import _PROJECTS
from autumn.core.utils.git import get_git_branch, get_git_hash
from autumn.core.utils.runs import build_run_id
from autumn.core.runs import ManagedRun
from autumn.core.utils.timer import Timer
from autumn.settings import BASE_PATH, DOCS_PATH, MODELS_PATH, OUTPUT_DATA_PATH, Region, Models
from summer.derived_outputs import DerivedOutputRequest
from summer.model import CompartmentalModel

from .params import ParameterSet, Params

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

ModelBuilder = Callable[[dict, dict], CompartmentalModel]


class LocalTaskRunner:
    def __init__(self, project):
        self.project = project

    def calibrate(self, num_chains: int, runtime: str, trigger=False, burn_in=500, samples=100):
        from autumn.infrastructure.tasks.calibrate import calibrate_task

        assert isinstance(runtime, str), "Runtime must be supplied as HMS strings"
        runtime_s = pd.to_timedelta(runtime) / timedelta(seconds=1)
        run_id = self.project._gen_run_id()

        logger.info(f"Calibrating {num_chains} chains for {runtime} with run_id {run_id}")
        calibrate_task(run_id, runtime_s, num_chains, False, "local")

        if trigger:
            logger.info(f"Triggering full run for {run_id}")
            self.full_run(run_id, burn_in, samples)
            self.powerbi(run_id)

        return ManagedRun(run_id)

    def full_run(self, run_id: str, burn_in: int, samples: int):
        from autumn.infrastructure.tasks.full import full_model_run_task

        full_model_run_task(run_id, burn_in, samples, False, "local")
        return ManagedRun(run_id)

    def powerbi(self, run_id: str):
        from autumn.infrastructure.tasks.powerbi import powerbi_task

        powerbi_task(run_id, "mle", False, "local")

        return ManagedRun(run_id)
    
    def generic(self, task_key, task_kwargs=None):
        from autumn.infrastructure.tasks.generic import generic_task

        run_id = self.project._gen_run_id()
        logger.info(f"Running {task_key}")

        generic_task(run_id, task_key=task_key, task_kwargs=task_kwargs, store="local")

        return ManagedRun(run_id)


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
        post_diff_output_requests: Dict[str, Dict] = None,
        ts_set: dict = None,
    ):
        self.region_name = region_name
        self.model_name = model_name
        self.build_model = build_model
        self.param_set = param_set
        self.plots = plots or {}
        self.calibration = calibration
        self.diff_output_requests = diff_output_requests or []
        self.post_diff_output_requests = post_diff_output_requests or {}
        self.ts_set = ts_set or None
        self.tasks = LocalTaskRunner(self)

        self._model = None
        self._scenario_models = None
        self._is_calibrating = False
        self._cal_params = [
            p["param_name"]
            for p in calibration.all_priors
            if not p["param_name"].startswith("random_process")
        ]
        if any(
            [p["param_name"].startswith("random_process.delta_values") for p in calibration.all_priors]
        ):
            self._cal_params.append("random_process.delta_values")

    def _calibrate(self, max_seconds: float, chain_idx: int, num_chains: int):
        """
        Calibrate the model using the baseline parameters.
        """
        with Timer(f"Running calibration for {self.model_name} {self.region_name}."):
            self.calibration.run(self, max_seconds, chain_idx, num_chains)

    def _start_calibrating(self):
        self._is_calibrating = True
        self._model = None

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
        if self._model is None:
            model = self.build_model(params_dict, build_options)
            if derived_outputs_whitelist:
                # Only calculate required derived outputs.
                model.set_derived_outputs_whitelist(derived_outputs_whitelist)
            if isinstance(model, CompartmentalModel2):
                self._model = model
                self._runner = self._initialize_model(model, params_dict, True)
                return self._runner.model
            else:
                model.run()
                return model
        else:
            pdict_exp = expand_nested_dict(params_dict)
            self._runner.run(pdict_exp)
            return self._runner.model

    def run_scenario_models(
        self,
        baseline_model: CompartmentalModel,
        scenario_params: List[Params],
        build_options: Optional[List[dict]] = None,
    ) -> List[CompartmentalModel]:
        """
        Runs all the project's scenarios with the given parameters.
        Returns the completed scenario models.
        """
        if build_options is None:
            build_options = [None] * len(scenario_params)

        models = []
        assert baseline_model.outputs is not None, "Baseline model has not been run yet."

        if isinstance(baseline_model, CompartmentalModel2):
            if self._scenario_models is None:
                self._scenario_models = {}
                for model_idx, params in enumerate(scenario_params):
                    params_dict = params.to_dict()
                    model = self.build_model(params_dict, None)
                    if model.times[0] != baseline_model.times[0]:
                        raise ValueError("Scenario start times must match baseline start times")
                    runner = self._initialize_model(model, params_dict, False)
                    self._scenario_models[model_idx] = runner
            for model_idx, params in enumerate(scenario_params):
                pdict_exp = expand_nested_dict(params.to_dict())
                self._scenario_models[model_idx].run(pdict_exp)
                models.append(self._scenario_models[model_idx].model)
        else:
            for model_idx, (params, build_opt) in enumerate(zip(scenario_params, build_options)):

                params_dict = params.to_dict()

                model = self.build_model(params_dict, build_opt)

                if model.times[0] != baseline_model.times[0]:
                    raise ValueError("Scenario start times must match baseline start times")

                self._run_model(model)
                models.append(model)

        return models

    def _initialize_model(self, model: CompartmentalModel2, params_dict, run=True):
        model.finalize()
        pdict_filt = {
            k: v
            for k, v in expand_nested_dict(params_dict).items()
            if k in model.get_input_parameters()
        }
        runner = model.get_runner(pdict_filt, dyn_params=self._cal_params)
        if run:
            runner.run(pdict_filt)
        return runner

    def _run_model(self, model: CompartmentalModel):
        """
        Run the model.
        """
        model.run()

    def get_path(self) -> Path:
        """
        Return a pathlib.Path to the current project directory
        """
        return Path(BASE_PATH) / "/".join(self._get_path().split(".")[:-1])

    def __repr__(self):
        return f"Project<{self.model_name}, {self.region_name}>"

    def _get_path(self):
        return _PROJECTS[self.model_name][self.region_name]

    def _gen_run_id(self):
        return build_run_id(self.model_name, self.region_name, get_git_hash()[0:8])


LOADED_PROJECTS = set()


def get_project(model_name: str, project_name: str, reload=False) -> Project:
    """
    Returns a project
    """
    # If a school closure project is requested, will call the relevant project builder function
    if model_name == Models.SM_COVID: # and project_name in Region.SCHOOL_PROJECT_REGIONS:
        # Ugly import within function definition to avoid circular imports
        from autumn.projects.sm_covid.common_school.project_maker import get_school_project

        project = get_school_project(project_name)
    elif model_name == Models.SM_COVID2: # and project_name in Region.SCHOOL_PROJECT_REGIONS:
        # Ugly import within function definition to avoid circular imports
        from autumn.projects.sm_covid2.common_school.project_maker import get_school_project

        project = get_school_project(project_name)

    # Otherwise, the project is loaded from the relevant project.py file
    else:
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
            sc_models = project.run_scenario_models(baseline_model, project.param_set.scenarios)
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
    calculate_post_diff_outputs(models, project.post_diff_output_requests)

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


def calculate_post_diff_outputs(
    models: List[CompartmentalModel], post_diff_output_requests: Dict[str, Dict]
):
    """
    Calculate outputs based on previously-computed differential outputs

    Models list assumed to be in order [baseline, sc1, sc2, ..., scN]

    post_diff_output_requests has form
    {
        "<output name>": {
            'sources': List[str],
            'func': Callable
        },
        ...
    }
    """
    for new_output_name, func_details in post_diff_output_requests.items():
        for model in models:
            calculated_sources = [model.derived_outputs[s] for s in func_details["sources"]]
            model.derived_outputs[new_output_name] = func_details["func"](*calculated_sources)
