import os
import logging

import luigi


from . import utils
from . import settings

logger = logging.getLogger(__name__)


class RunDHHS(utils.BaseTask):
    """DHHS post processing"""

    commit = luigi.Parameter()  # Unique run id string

    def get_path(self):
        return os.path.expanduser("~/hello.txt")

    def output(self):
        return luigi.LocalTarget(self.get_path())

    def safe_run(self):
        logger.info("Hello World %s", self.commit)
        with open(self.get_path(), "w") as f:
            f.write("Hello World")

        logger.info("Goodbye World")
