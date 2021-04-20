"""
A tool for application models to register themselves so that they serve a standard interface
"""
import logging
import os
from datetime import datetime
from typing import Callable, Dict, List

import yaml
from summer.derived_outputs import DerivedOutputRequest

from autumn import db
from autumn.utils.params import load_params, load_targets
from autumn.utils.scenarios import Scenario, calculate_differential_outputs, get_scenario_start_index
from autumn.utils.utils import get_git_branch, get_git_hash
from settings import OUTPUT_DATA_PATH
from utils.timer import Timer

logger = logging.getLogger(__name__)


class AppRegion:
    """
    A region that is simulated by the app.
    This class knows specificially how to apply the app to this particular region.

    Eg. we could have a COVID-19 model, with an `app_name` of "covid_19" which simulates
        the state of Victoria, which would have a `region_name` of "victoria".
    """

    def __init__(self, app_name: str, region_name: str, build_model, calibrate_model):
        self.region_name = region_name
        self.app_name = app_name
        self._build_model = build_model
        self._calibrate_model = calibrate_model

    @property
    def params(self):
        """Returns the model parameters for the given region"""
        return load_params(self.app_name, self.region_name)

    @property
    def targets(self):
        """Returns the calibration targets for the given region"""
        return load_targets(self.app_name, self.region_name)

    def calibrate_model(self, max_seconds: int, run_id: int, num_chains: int):
        """Runs a calibtation for the given region"""
        return self._calibrate_model(max_seconds, run_id, num_chains)

    def build_model(self, params: dict):
        """Builds the SUMMER model for the given region"""
        return self._build_model(params)

    def run_model(self, run_scenarios=True):
        """
        Runs the SUMMER model for the given region, storing the outputs on disk.
        This is the primary entry point for the App code; clients should use
        build_and_run_scenarios directly if they have special I/O requirements
        """
        logger.info(f"Running {self.app_name} {self.region_name}...")
        params = self.params

        # Ensure project folder exists.
        project_dir = os.path.join(OUTPUT_DATA_PATH, "run", self.app_name, self.region_name)
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        output_dir = os.path.join(project_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)

        # Determine where to save model outputs
        output_db_path = os.path.join(output_dir, "outputs")

        # Save model parameters to output dir.
        param_path = os.path.join(output_dir, "params.yml")
        with open(param_path, "w") as f:
            yaml.dump(params, f)

        # Save model run metadata to output dir.
        meta_path = os.path.join(output_dir, "meta.yml")
        metadata = {
            "model_name": self.app_name,
            "region_name": self.region_name,
            "start_time": timestamp,
            "git_branch": get_git_branch(),
            "git_commit": get_git_hash(),
        }
        with open(meta_path, "w") as f:
            yaml.dump(metadata, f)

        with Timer("Running model scenarios"):
            scenarios = self.build_and_run_scenarios(run_scenarios)

        with Timer("Saving model outputs to the database"):
            outputs_db = db.FeatherDatabase(output_db_path)
            all_outputs = self.process_scenario_outputs(scenarios)
            db.store.save_model_outputs(outputs_db, **all_outputs)

    def build_and_run_scenarios(self, run_scenarios=True, update_func: Callable[[dict], dict]=None) -> List[Scenario]:
        """
        Construct and run the scenarios (as specified in self.params["scenarios"]), and return
        the list of Scenario objects on completion

        Args:
            run_scenarios: Switch specifying whether to run all scenarios (or if False, only the baseline)
            update_func: Optional function to update the parameter set for each run

        Returns:
            List of constructed scenarios, whose .model members will contain the output data of the run

        """

        params = self.params

        num_scenarios = 1 + len(params["scenarios"].keys())
        scenarios = []
        for scenario_idx in range(num_scenarios):
            scenario = Scenario(self._build_model, scenario_idx, params)
            scenarios.append(scenario)

        # Run the baseline scenario.
        baseline_scenario = scenarios[0]
        baseline_scenario.run(update_func=update_func)
        baseline_model = baseline_scenario.model

        if not run_scenarios:
            # Do not run non-baseline models
            scenarios = scenarios[:1]

        # Run all the other scenarios
        for scenario in scenarios[1:]:
            scenario.run(base_model=baseline_model, update_func=update_func)

        return scenarios

    def process_scenario_outputs(self, scenarios: List[Scenario], run_id: int=0, chain_id: int=None) -> dict:
        """Do any required postprocessing of scenario outputs (particularly those that require comparison
        to baseline)
        
        Args:
            scenarios (List[Scenario]): List of (run) scenarios, as returned from build_and_run_scenarios
            run_id (int, optional): Required for multiple (usually remote) runs, not required for single local runs
            chain_id (int, optional): Used by MCMC runs, not required for single runs
        
        Returns:
            dict: Dict whose keys are our expected table names, and values are the (processed) DataFrames
        """

        models = [s.model for s in scenarios]
        calculate_differential_outputs(models, self.targets)

        # Adjustments for scenario outputs to align with baseline
        baseline = scenarios[0]
        # FIXME: Accessing private member of model class; prefer not to modify summer code just for this
        cum_out_keys = [k for k, req in baseline.model._derived_output_requests.items() \
            if req['request_type'] == DerivedOutputRequest.CUMULATIVE and req['save_results'] == True]

        for scenario in scenarios[1:]:
            baseline_start_index = get_scenario_start_index(baseline.model.times, scenario.model.times[0])

            # Adjust cumulative outputs to start at baseline value rather than 0
            for output_key in cum_out_keys:
                baseline_offset = baseline.model.derived_outputs[output_key][baseline_start_index]
                scenario.model.derived_outputs[output_key] += baseline_offset

        #Build and store outputs
        outputs_df = db.store.build_outputs_table(models, run_id=run_id, chain_id=chain_id)
        derived_outputs_df = db.store.build_derived_outputs_table(models, run_id=run_id, chain_id=chain_id)

        processed_out = {
            db.store.Table.OUTPUTS: outputs_df,
            db.store.Table.DERIVED: derived_outputs_df
        }

        return processed_out


class App:
    """
    A disease model which may be applied to multiple regions.

    Eg. we could have a COVID-19 model, with an `app_name` of "covid_19" which simulates
        the state of Victoria, which would have a `region_name` of "victoria".
    """

    def __init__(self, app_name):
        self.app_name = app_name
        self.region_names = []
        self.region_modules = {}

    def register(self, app_region):
        assert app_region.region_name not in self.region_names
        assert app_region.app_name == self.app_name
        self.region_names.append(app_region.region_name)
        self.region_modules[app_region.region_name] = app_region

    def get_region(self, region_name: str) -> AppRegion:
        return self.region_modules[region_name]
