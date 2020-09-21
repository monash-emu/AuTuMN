import os
import glob
import shutil
import logging
from tempfile import TemporaryDirectory

import luigi

from autumn.tool_kit import Timer
from autumn.inputs import build_input_database
from autumn.inputs.database import input_db_path
from autumn.constants import OUTPUT_DATA_PATH
from autumn import plots, db

from . import utils
from . import settings

logger = logging.getLogger(__name__)

# TODO: Upload prior plots, params and calibration config
# TODO: Upload intermediate plots and databases?

MLE_PARAMS_PATH = os.path.join(settings.BASE_DIR, "data", "calibration_outputs", "mle-params.yml")


class RunCalibrate(luigi.Task):
    """Master task, requires all other tasks"""

    run_id = luigi.Parameter()  # Unique run id string
    runtime = luigi.IntParameter()  # Runtime in seconds
    num_chains = luigi.IntParameter()

    def requires(self):
        """
        By the completion of this pipeline we want:
            - all logs uploaded to S3
            - all output data uploaded to S3
            - all plots uploaded to S3
        """
        upload_db_tasks = [
            UploadDatabaseTask(
                run_id=self.run_id, chain_id=i, num_chains=self.num_chains, runtime=self.runtime
            )
            for i in range(self.num_chains)
        ]
        upload_plots_task = UploadPlotsTask(
            run_id=self.run_id, num_chains=self.num_chains, runtime=self.runtime
        )
        upload_params_task = UploadParamsTask(
            run_id=self.run_id, num_chains=self.num_chains, runtime=self.runtime
        )
        return [*upload_db_tasks, upload_plots_task, upload_params_task]


class BuildInputDatabaseTask(utils.BaseTask):
    """Builds the input database"""

    def output(self):
        return luigi.LocalTarget(input_db_path)

    def safe_run(self):
        build_input_database()


class CalibrationChainTask(utils.ParallelLoggerTask):
    """Runs the calibration for a single chain"""

    run_id = luigi.Parameter()  # The calibration to run
    runtime = luigi.IntParameter()  # Runtime in seconds
    chain_id = luigi.IntParameter()  # Unique chain id
    num_chains = luigi.IntParameter()

    def requires(self):
        paths = ["logs/calibrate", "data/calibration_outputs"]
        dir_tasks = [utils.BuildLocalDirectoryTask(dirname=p) for p in paths]
        return [BuildInputDatabaseTask(), *dir_tasks]

    def output(self):
        return luigi.LocalTarget(self.get_output_db_path())

    def get_log_filename(self):
        return f"calibrate/run-{self.chain_id}.log"

    def safe_run(self):
        app_region = utils.get_app_region(self.run_id)
        msg = f"Running {app_region.app_name} {app_region.region_name} calibration with chain id {self.chain_id} with runtime {self.runtime}s"
        with Timer(msg):
            # Run the calibration
            app_region.calibrate_model(self.runtime, self.chain_id, self.num_chains)

        # Place the completed chain database in the correct output folder
        src_db_path = self.find_src_db_path()
        dest_db_path = self.get_output_db_path()
        logger.info(
            f"Copying output database for chain id {self.chain_id} from {src_db_path} to {dest_db_path}"
        )
        shutil.move(src_db_path, dest_db_path)

    def get_output_db_path(self):
        return os.path.join(
            OUTPUT_DATA_PATH, "remote", "data", "calibration_outputs", self.get_filename()
        )

    def find_src_db_path(self):
        filename = self.get_filename()
        calibrate_glob = os.path.join(OUTPUT_DATA_PATH, "calibrate", "**", self.get_filename())
        src_paths = glob.glob(calibrate_glob, recursive=True)
        assert len(src_paths) < 2, f"Multiple output databases found for {filename}"
        assert len(src_paths) > 0, f"No output databases found for {filename}"
        return src_paths[0]

    def get_filename(self):
        return utils.get_calibration_db_filename(self.chain_id)


class UploadDatabaseTask(utils.UploadS3Task):

    chain_id = luigi.IntParameter()  # Unique chain id
    runtime = luigi.IntParameter()  # Runtime in seconds
    num_chains = luigi.IntParameter()  # The number of chains to run

    def requires(self):
        return CalibrationChainTask(
            run_id=self.run_id,
            chain_id=self.chain_id,
            num_chains=self.num_chains,
            runtime=self.runtime,
        )

    def get_src_path(self):
        filename = utils.get_calibration_db_filename(self.chain_id)
        return os.path.join(settings.BASE_DIR, "data", "calibration_outputs", filename)


class PlotOutputsTask(utils.BaseTask):
    """Plots the database outputs"""

    run_id = luigi.Parameter()
    runtime = luigi.IntParameter()  # Runtime in seconds
    num_chains = luigi.IntParameter()  # The number of chains to run

    def requires(self):
        paths = ["plots"]
        dir_tasks = [utils.BuildLocalDirectoryTask(dirname=p) for p in paths]
        chain_tasks = [
            CalibrationChainTask(
                run_id=self.run_id, chain_id=i, num_chains=self.num_chains, runtime=self.runtime
            )
            for i in range(self.num_chains)
        ]
        return [*dir_tasks, *chain_tasks]

    def output(self):
        target_file = os.path.join(settings.BASE_DIR, "plots", "loglikelihood-traces.png")
        return luigi.LocalTarget(target_file)

    def safe_run(self):
        mcmc_dir = os.path.join(settings.BASE_DIR, "data", "calibration_outputs")
        plot_dir = os.path.join(settings.BASE_DIR, "plots")
        app_region = utils.get_app_region(self.run_id)
        plots.calibration.plot_post_calibration(app_region.targets, mcmc_dir, plot_dir)


class UploadPlotsTask(utils.UploadS3Task):
    """Uploads output plots"""

    run_id = luigi.Parameter()
    runtime = luigi.IntParameter()  # Runtime in seconds
    num_chains = luigi.IntParameter()  # The number of chains to run

    def requires(self):
        return PlotOutputsTask(run_id=self.run_id, runtime=self.runtime, num_chains=self.num_chains)

    def get_src_path(self):
        return os.path.join(settings.BASE_DIR, "plots")


class GetParamsTask(utils.BaseTask):
    """Get MLE parameters from final database"""

    run_id = luigi.Parameter()
    runtime = luigi.IntParameter()  # Runtime in seconds
    num_chains = luigi.IntParameter()  # The number of chains to run

    def requires(self):
        return [
            CalibrationChainTask(
                run_id=self.run_id, chain_id=i, num_chains=self.num_chains, runtime=self.runtime
            )
            for i in range(self.num_chains)
        ]

    def output(self):
        return luigi.LocalTarget(MLE_PARAMS_PATH)

    def safe_run(self):
        # Collate all database's MCMC runs and params into a single db
        mcmc_dir = os.path.join(settings.BASE_DIR, "data", "calibration_outputs")

        database_paths = [
            os.path.join(mcmc_dir, dbname)
            for dbname in os.listdir(mcmc_dir)
            if "chain" in dbname and dbname.endswith(".db")
        ]
        with TemporaryDirectory() as tmp_dir_path:
            collated_db_path = os.path.join(tmp_dir_path, "collated.db")
            db.process.collate_databases(
                database_paths, collated_db_path, tables=["mcmc_run", "mcmc_params"]
            )
            db.store.save_mle_params(collated_db_path, MLE_PARAMS_PATH)


class UploadParamsTask(utils.UploadS3Task):
    """Uploads MLE parameters"""

    run_id = luigi.Parameter()
    runtime = luigi.IntParameter()  # Runtime in seconds
    num_chains = luigi.IntParameter()  # The number of chains to run

    def requires(self):
        return GetParamsTask(run_id=self.run_id, num_chains=self.num_chains, runtime=self.runtime)

    def get_src_path(self):
        return MLE_PARAMS_PATH
