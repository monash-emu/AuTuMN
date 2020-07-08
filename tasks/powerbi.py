import os
import logging

import luigi

from autumn.tool_kit import Timer
from autumn.inputs import build_input_database
from autumn.inputs.database import input_db_path
from autumn.db import models
from autumn.tool_kit.uncertainty import (
    add_uncertainty_weights,
    add_uncertainty_quantiles,
)

from . import utils
from . import settings

logger = logging.getLogger(__name__)

PRUNED_DIR = os.path.join(settings.BASE_DIR, "data/powerbi/pruned/")
COLLATED_DB_PATH = os.path.join(settings.BASE_DIR, "data/powerbi/collated.db")
COLLATED_PRUNED_DB_PATH = os.path.join(settings.BASE_DIR, "data/powerbi/collated-pruned.db")
UNCERTAINTY_OUTPUTS = [
    "incidence",
    "notifications",
    "infection_deathsXall",
    "prevXlateXclinical_icuXamong",
]


def get_final_db_path(run_id: str):
    return os.path.join(settings.BASE_DIR, "data", "powerbi", f"powerbi-{run_id}.db")


class RunPowerBI(luigi.Task):
    """Master task, requires all other tasks"""

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        return UploadDatabaseTask()


class BuildInputDatabaseTask(utils.BaseTask):
    """Builds the input database"""

    def output(self):
        return luigi.LocalTarget(input_db_path)

    def safe_run(self):
        build_input_database()


class UncertaintyWeightsTask(utils.ParallelLoggerTask):
    """Calculates uncertainty weights for each database"""

    def requires(self):
        download_task = utils.DownloadS3Task(run_id=self.run_id, src_path=self.get_src_db_relpath())
        paths = ["logs/powerbi", "data/powerbi"]
        dir_tasks = [utils.BuildLocalDirectoryTask(dirname=p) for p in paths]
        return [BuildInputDatabaseTask(), download_task, *dir_tasks]

    def safe_run(self):
        add_uncertainty_weights(output_name, db_path)


class PruneFullRunDatabaseTask(utils.ParallelLoggerTask):
    """Prunes unneeded data for each full model run db"""

    chain_id = luigi.IntParameter()  # Unique chain id

    def requires(self):
        return [UncertaintyWeightsTask]  # etc etc

    def get_dest_path(self):
        return os.path.join(PRUNED_DIR, f"pruned-{self.chain_idx}.db")

    def output(self):
        return luigi.LocalTarget(self.get_dest_path())

    def safe_run(self):
        src_filename = utils.get_full_model_run_db_filename(self.chain_id)
        src_db_path = os.path.join(settings.BASE_DIR, "data", "full_model_runs", src_filename)
        models.prune(src_db_path, self.get_dest_path())


class CollationTask(utils.BaseTask):
    """Collates each full model run db into one."""

    def requires(self):
        key_prefix = os.path.join(self.run_id, "data/full_model_runs")
        chain_db_keys = utils.list_s3(key_prefix, key_suffix=".db")
        chain_db_idxs = [int(k.replace(".db", "").split("_")[-1]) for k in chain_db_keys]
        return [PruneFullRunDatabaseTask(chain_id=i) for i in chain_idxs]

    def output(self):
        return luigi.LocalTarget(COLLATED_DB_PATH)

    def safe_run(self):
        models.collate_databases(PRUNED_DIR, COLLATED_DB_PATH)


class CalculateUncertaintyTask(utils.BaseTask):
    """
    Calculates uncertainty for model outputs and 
    prunes collated database and unpiovot data into PowerBI friendly form.
    """

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        return CollationTask()

    def output(self):
        return luigi.LocalTarget(COLLATED_PRUNED_DB_PATH)

    def safe_run(self):
        add_uncertainty_quantiles(COLLATED_DB_PATH)
        models.prune(COLLATED_DB_PATH, COLLATED_PRUNED_DB_PATH)


class UnpivotTask(utils.BaseTask):
    """Unpiovots data into PowerBI friendly form."""

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        return CalculateUncertaintyTask()

    def output(self):
        return luigi.LocalTarget(get_final_db_path(self.run_id))

    def safe_run(self):
        models.unpivot(COLLATED_PRUNED_DB_PATH, get_final_db_path(self.run_id))


class UploadDatabaseTask(utils.UploadS3Task):

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        return UnpivotTask()

    def get_src_path(self):
        return get_final_db_path(self.run_id)

