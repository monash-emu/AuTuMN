import os
import logging

import luigi

from autumn.tool_kit import Timer
from autumn.inputs import build_input_database
from autumn.inputs.database import input_db_path
from autumn.constants import OUTPUT_DATA_PATH
from autumn import db, plots

from . import utils
from . import settings

logger = logging.getLogger(__name__)

PRUNED_DIR = os.path.join(settings.BASE_DIR, "data/powerbi/pruned/")
COLLATED_DB_PATH = os.path.join(settings.BASE_DIR, "data/powerbi/collated.db")
COLLATED_PRUNED_DB_PATH = os.path.join(settings.BASE_DIR, "data/powerbi/collated-pruned.db")


def get_final_db_path(run_id: str):
    run_slug = run_id.replace("/", "-")
    return os.path.join(settings.BASE_DIR, "data", "powerbi", f"powerbi-{run_slug}.db")


class RunPowerBI(luigi.Task):
    """Master task, requires all other tasks"""

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        return UploadPlotsTask(run_id=self.run_id)


class BuildInputDatabaseTask(utils.BaseTask):
    """Builds the input database"""

    def output(self):
        return luigi.LocalTarget(input_db_path)

    def safe_run(self):
        build_input_database()


class PruneFullRunDatabaseTask(utils.ParallelLoggerTask):
    """Prunes unneeded data for each full model run db"""

    run_id = luigi.Parameter()  # Unique run id string
    chain_id = luigi.IntParameter()  # Unique chain id

    def requires(self):
        src_filename = utils.get_full_model_run_db_filename(self.chain_id)
        src_db_relpath = os.path.join("data", "full_model_runs", src_filename)
        return [
            BuildInputDatabaseTask(),
            utils.DownloadS3Task(run_id=self.run_id, src_path=src_db_relpath),
            utils.BuildLocalDirectoryTask(dirname="logs/powerbi"),
            utils.BuildLocalDirectoryTask(dirname="data/powerbi/pruned"),
        ]

    def get_dest_path(self):
        return os.path.join(PRUNED_DIR, f"pruned-{self.chain_id}.db")

    def output(self):
        return luigi.LocalTarget(self.get_dest_path())

    def safe_run(self):
        with Timer(f"Pruning database for chain {self.chain_id}"):
            src_filename = utils.get_full_model_run_db_filename(self.chain_id)
            src_db_path = os.path.join(settings.BASE_DIR, "data", "full_model_runs", src_filename)
            dest_db_path = self.get_dest_path()
            db.process.prune_chain(src_db_path, dest_db_path)

    def get_log_filename(self):
        return f"powerbi/prune-{self.chain_id}.log"


class CollationTask(utils.BaseTask):
    """Collates each full model run db into one."""

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        key_prefix = os.path.join(self.run_id, "data/full_model_runs")
        chain_db_keys = utils.list_s3(key_prefix, key_suffix=".db")
        chain_db_idxs = [int(k.replace(".db", "").split("_")[-1]) for k in chain_db_keys]
        return [PruneFullRunDatabaseTask(run_id=self.run_id, chain_id=i) for i in chain_db_idxs]

    def output(self):
        return luigi.LocalTarget(COLLATED_DB_PATH)

    def safe_run(self):
        with Timer(f"Collating databases"):
            src_db_paths = [
                os.path.join(PRUNED_DIR, fname)
                for fname in os.listdir(PRUNED_DIR)
                if fname.endswith(".db")
            ]
            db.process.collate_databases(src_db_paths, COLLATED_DB_PATH)


class CalculateUncertaintyTask(utils.BaseTask):
    """
    Calculates uncertainty for model outputs and 
    prunes collated database and unpiovot data into PowerBI friendly form.
    """

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        return CollationTask(run_id=self.run_id)

    def output(self):
        return luigi.LocalTarget(COLLATED_PRUNED_DB_PATH)

    def safe_run(self):
        app_region = utils.get_app_region(self.run_id)
        with Timer(f"Calculating uncertainty quartiles"):
            db.uncertainty.add_uncertainty_quantiles(COLLATED_DB_PATH, app_region.targets)

        with Timer(f"Pruning final database"):
            db.process.prune_final(COLLATED_DB_PATH, COLLATED_PRUNED_DB_PATH)


class UnpivotTask(utils.BaseTask):
    """Unpiovots data into PowerBI friendly form."""

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        return CalculateUncertaintyTask(run_id=self.run_id)

    def output(self):
        return luigi.LocalTarget(get_final_db_path(self.run_id))

    def safe_run(self):
        with Timer(f"Unpivoting final database"):
            db.process.unpivot(COLLATED_PRUNED_DB_PATH, get_final_db_path(self.run_id))


class UploadDatabaseTask(utils.UploadS3Task):

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        return UnpivotTask(run_id=self.run_id)

    def get_src_path(self):
        return get_final_db_path(self.run_id)


class PlotUncertaintyTask(utils.BaseTask):
    """Plots the database output uncertainty"""

    run_id = luigi.Parameter()

    def requires(self):
        return UploadDatabaseTask(run_id=self.run_id)

    def output(self):
        app_region = utils.get_app_region(self.run_id)
        target_names = [t["output_key"] for t in app_region.targets.values() if t.get("quantiles")]
        target_files = [
            os.path.join(settings.BASE_DIR, "plots", "uncertainty", o, f"uncertainty-{o}-0.png",)
            for o in target_names
        ]
        return [luigi.LocalTarget(f) for f in target_files]

    def safe_run(self):
        app_region = utils.get_app_region(self.run_id)
        plot_dir = os.path.join(settings.BASE_DIR, "plots", "uncertainty")
        db_path = get_final_db_path(self.run_id)
        plots.uncertainty.plot_uncertainty(app_region.targets, db_path, plot_dir)


class UploadPlotsTask(utils.UploadS3Task):
    """Uploads uncertainty plots"""

    run_id = luigi.Parameter()

    def requires(self):
        return PlotUncertaintyTask(run_id=self.run_id)

    def get_src_path(self):
        return os.path.join(settings.BASE_DIR, "plots", "uncertainty")
