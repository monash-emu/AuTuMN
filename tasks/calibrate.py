import os
import glob
import shutil
import logging

import luigi
from luigi.contrib.s3 import S3Target

from autumn.tool_kit import Timer
from autumn.inputs import build_input_database
from autumn.inputs.database import input_db_path
from autumn.plots.database_plots import plot_from_mcmc_databases
from autumn.constants import OUTPUT_DATA_PATH
from apps.covid_19.calibration import get_calibration_func

from . import utils
from . import settings

logger = logging.getLogger(__name__)

# TODO: Upload prior plots, params and calibration config
# TODO: Upload intermediate plots and databases?

BASE_DIR = os.path.join(OUTPUT_DATA_PATH, "remote")


class RunCalibrate(luigi.Task):
    """Master task, requires all other tasks"""

    run_id = luigi.Parameter()  # Unique run id string
    num_chains = luigi.IntParameter()

    def requires(self):
        """
        By the completion of this pipeline we want:
            - all logs uploaded to S3
            - all output data uploaded to S3
            - all plots uploaded to S3
        """
        upload_db_tasks = [self.get_upload_db_task(i) for i in range(self.num_chains)]
        upload_log_tasks = [self.get_upload_log_task(i) for i in range(self.num_chains)]
        upload_plots_task = UploadPlotsTask(
            num_chains=self.num_chains,
            src_path=os.path.join(BASE_DIR, "plots"),
            dest_key=os.path.join(self.run_id, "plots"),
            bucket=settings.S3_BUCKET,
        )
        return [*upload_log_tasks, *upload_db_tasks, upload_plots_task]

    def get_upload_db_task(self, chain_id):
        filename = f"outputs_calibration_chain_{chain_id}.db"
        return UploadDatabaseTask(
            chain_id=chain_id,
            src_path=os.path.join(BASE_DIR, "data", filename),
            dest_key=os.path.join(self.run_id, "data", "calibration_outputs", filename),
            bucket=settings.S3_BUCKET,
        )

    def get_upload_log_task(self, chain_id):
        filename = f"run-{chain_id}.log"
        return UploadLogs(
            chain_id=chain_id,
            src_path=os.path.join(BASE_DIR, "logs", filename),
            dest_key=os.path.join(self.run_id, "logs", filename),
            bucket=settings.S3_BUCKET,
        )


class BuildInputDatabaseTask(luigi.Task):
    """Builds the input database"""

    def output(self):
        return luigi.LocalTarget(input_db_path)

    def run(self):
        build_input_database()


class CalibrationChainTask(luigi.Task):
    """Runs the calibration for a single chain"""

    model_name = luigi.Parameter()  # The calibration to run
    runtime = luigi.IntParameter()  # Runtime in seconds
    chain_id = luigi.IntParameter()  # Unique chain id

    def requires(self):
        paths = ["logs", "data", "plots"]
        dir_tasks = [utils.BuildLocalDirectoryTask(dirname=p, base_path=BASE_DIR) for p in paths]
        return [BuildInputDatabaseTask(), *dir_tasks]

    def output(self):
        return luigi.LocalTarget(self.get_output_db_path())

    def run(self):
        logfile_path = os.path.join(BASE_DIR, "logs", f"run-{self.chain_id}.log")
        log_format = "%(asctime)s %(module)s:%(levelname)s: %(message)s"
        logger_names = ["autumn", "summer"]
        for logger_name in logger_names:
            formatter = logging.Formatter(log_format)
            handler = logging.FileHandler(logfile_path)
            handler.setFormatter(formatter)
            _logger = logging.getLogger(logger_name)
            _logger.handlers = []
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)

        msg = f"Running {self.model_name} calibration with chain id {self.chain_id} with runtime {self.runtime}s"
        with Timer(msg):
            # Run the calibration
            calibrate_func = get_calibration_func(self.model_name)
            calibrate_func(self.runtime, self.chain_id)

        # Place the completed chain database in the correct output folder
        src_db_path = self.find_src_db_path()
        dest_db_path = self.get_output_db_path()
        logger.info(
            f"Copying output database for chain id {self.chain_id} from {src_db_path} to {dest_db_path}"
        )
        shutil.move(src_db_path, dest_db_path)

    def get_output_db_path(self):
        return os.path.join(OUTPUT_DATA_PATH, "remote", "data", self.get_filename())

    def find_src_db_path(self):
        filename = self.get_filename()
        calibrate_glob = os.path.join(OUTPUT_DATA_PATH, "calibrate", "**", self.get_filename())
        src_paths = glob.glob(calibrate_glob, recursive=True)
        assert len(src_paths) < 2, f"Multiple output databases found for {filename}"
        assert len(src_paths) > 0, f"No output databases found for {filename}"
        return src_paths[0]

    def get_filename(self):
        return f"outputs_calibration_chain_{self.chain_id}.db"


# Handle failure - upload logs
CalibrationChainTask.event_handler(luigi.Event.FAILURE)(utils.upload_logs_on_failure)


class UploadDatabaseTask(utils.UploadFileS3Task):

    chain_id = luigi.IntParameter()  # Unique chain id

    def requires(self):
        return CalibrationChainTask(chain_id=self.chain_id)


class UploadLogs(utils.UploadFileS3Task):
    """Uploads the log files for a given task"""

    chain_id = luigi.IntParameter()  # Unique chain id

    def requires(self):
        return CalibrationChainTask(chain_id=self.chain_id)


class PlotOutputsTask(luigi.Task):
    """Plots the database outputs"""

    num_chains = luigi.IntParameter()  # The number of chains to run

    def requires(self):
        return [CalibrationChainTask(chain_id=i) for i in range(self.num_chains)]

    def output(self):
        target_file = os.path.join(BASE_DIR, "plots", "loglikelihood-traces.png")
        return luigi.LocalTarget(target_file)

    def run(self):
        mcmc_dir = os.path.join(BASE_DIR, "data")
        plot_dir = os.path.join(BASE_DIR, "plots")
        plot_from_mcmc_databases(mcmc_dir, plot_dir)


class UploadPlotsTask(utils.UploadFileS3Task):
    """Uploads output plots"""

    num_chains = luigi.IntParameter()  # The number of chains to run

    def requires(self):
        return PlotOutputsTask(self.num_chains)
