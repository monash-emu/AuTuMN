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
