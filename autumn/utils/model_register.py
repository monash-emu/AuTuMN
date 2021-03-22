"""
A tool for application models to register themselves so that they serve a standard interface
"""
import logging
import os
from datetime import datetime

import yaml

from autumn import db
from autumn.utils.params import load_params, load_targets
from autumn.utils.scenarios import Scenario
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
            num_scenarios = 1 + len(params["scenarios"].keys())
            scenarios = []
            for scenario_idx in range(num_scenarios):
                scenario = Scenario(self._build_model, scenario_idx, params)
                scenarios.append(scenario)

            # Run the baseline scenario.
            baseline_scenario = scenarios[0]
            baseline_scenario.run()
            baseline_model = baseline_scenario.model

            if not run_scenarios:
                # Do not run non-baseline models
                scenarios = scenarios[:1]

            # Run all the other scenarios
            for scenario in scenarios[1:]:
                scenario.run(base_model=baseline_model)

        with Timer("Saving model outputs to the database"):
            models = [s.model for s in scenarios]
            outputs_db = db.FeatherDatabase(output_db_path)
            outputs_df = db.store.build_outputs_table(models, run_id=0)
            derived_outputs_df = db.store.build_derived_outputs_table(models, run_id=0)
            outputs_db.dump_df(db.store.Table.OUTPUTS, outputs_df)
            outputs_db.dump_df(db.store.Table.DERIVED, derived_outputs_df)


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
