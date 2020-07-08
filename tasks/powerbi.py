import logging

import luigi


from . import utils
from . import settings

logger = logging.getLogger(__name__)


class RunPowerBI(luigi.Task):
    """Master task, requires all other tasks"""

    run_id = luigi.Parameter()  # Unique run id string

    def requires(self):
        return []


class BuildInputDatabaseTask(utils.BaseTask):
    """Builds the input database"""

    def output(self):
        return luigi.LocalTarget(input_db_path)

    def safe_run(self):
        build_input_database()


class UncertaintyWeightsTask(utils.ParallelLoggerTask):
    """Calculates uncertainty weights for each database"""

    pass


class PruneFullRunDatabaseTask(utils.ParallelLoggerTask):
    """Prunes unneeded data for each full model run db"""

    pass


class CalculateUncertaintyTask(utils.BaseTask):
    """Collates each full model run db into one and calculates uncertainty for model outputs."""

    pass


class UnpivotTask(utils.BaseTask):
    """Prune collated database and unpiovot data into PowerBI friendly form."""

    pass


class UploadDatabaseTask(utils.BaseTask):
    pass
