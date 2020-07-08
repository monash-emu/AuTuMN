import os
import logging

import luigi

from autumn.tool_kit import Timer
from autumn.inputs import build_input_database
from autumn.inputs.database import input_db_path
from apps.covid_19.calibration.base import run_full_models_for_mcmc

from . import utils
from . import settings

logger = logging.getLogger(__name__)


class RunFullModels(luigi.Task):
    """Master task, requires all other tasks"""

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        # Get number of uploaded dbs
        key_prefix = os.path.join(self.run_id, "data/calibration_outputs")
        chain_db_keys = utils.list_s3(key_prefix, key_suffix=".db")
        chain_db_idxs = [int(k.replace(".db", "").split("_")[-1]) for k in chain_db_keys]
        return [UploadDatabaseTask(run_id=self.run_id, chain_id=i) for i in chain_db_idxs]


class BuildInputDatabaseTask(utils.BaseTask):
    """Builds the input database"""

    def output(self):
        return luigi.LocalTarget(input_db_path)

    def safe_run(self):
        build_input_database()


class FullModelRunTask(utils.ParallelLoggerTask):
    """Runs the calibration for a single chain"""

    chain_id = luigi.IntParameter()  # Unique chain id
    model_name = luigi.Parameter()  # The calibration to run
    burn_in = luigi.IntParameter()

    def requires(self):
        # Download task
        download_task = utils.DownloadS3Task(run_id=self.run_id, src_path=self.get_src_db_relpath())
        paths = ["logs/full_model_runs", "data/full_model_runs"]
        dir_tasks = [utils.BuildLocalDirectoryTask(dirname=p) for p in paths]
        return [BuildInputDatabaseTask(), download_task, *dir_tasks]

    def output(self):
        return luigi.LocalTarget(self.get_output_db_path())

    def get_log_filename(self):
        return f"full_model_runs/run-{self.chain_id}.log"

    def safe_run(self):
        msg = f"Running {self.model_name} full model with burn-in of {self.burn_in}s"
        with Timer(msg):
            src_db_path = os.path.join(settings.BASE_DIR, self.get_src_db_relpath())
            run_full_models_for_mcmc(
                self.model_name, self.burn_in, src_db_path, self.get_output_db_path()
            )

    def get_src_db_relpath(self):
        filename = utils.get_calibration_db_filename(self.chain_id)
        return os.path.join("data", "calibration_outputs", filename)

    def get_output_db_path(self):
        filename = utils.get_full_model_run_db_filename(self.chain_id)
        return os.path.join(settings.BASE_DIR, "data", "full_model_runs", filename)


class UploadDatabaseTask(utils.UploadS3Task):

    chain_id = luigi.IntParameter()  # Unique chain id

    def requires(self):
        return FullModelRunTask(run_id=self.run_id, chain_id=self.chain_id)

    def get_src_path(self):
        filename = utils.get_full_model_run_db_filename(self.chain_id)
        return os.path.join(settings.BASE_DIR, "data", "full_model_runs", filename)

