"""
User-defined values for parameterized builds.
"""
import os
import logging

from . import buildkite

logger = logging.getLogger(__name__)


class BuildkiteParams:
    """Generic buildkite params"""

    def __init__(self):
        self.build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
        self.commit = os.environ["BUILDKITE_COMMIT"]
        self.branch = os.environ["BUILDKITE_BRANCH"]


class BaseParams:
    def __init__(self):
        self.buildkite = BuildkiteParams()
        self.env_run_id = os.environ.get("RUN_ID")
        if self.env_run_id:
            logger.info("Found run id from envar: %s", run_id)
        else:
            logger.info("Using user-supplied run name.")

    @property
    def run_id(self):
        if self.env_run_id:
            return self.env_run_id
        else:
            run_id = buildkite.get_metadata("run-id")
            if not run_id:
                raise ValueError("No user-supplied `run_id` found.")
            else:
                return run_id

    @property
    def trigger_downstream(self):
        """Whether we should trigger a downstream build."""
        trigger_str = buildkite.get_metadata("trigger-downstream") or "yes"
        return trigger_str == "yes"


class CalibrateParams(BaseParams):
    """Params for calibrating a model"""

    @property
    def model_name(self):
        """Name of the model to build."""
        return buildkite.get_metadata("model-name")

    @property
    def chains(self):
        """Number of parallel chains to run."""
        num_chains = buildkite.get_metadata("num-chains") or 7
        return int(num_chains)

    @property
    def runtime(self):
        """Number of seconds to run MCMC."""
        run_time_hours = buildkite.get_metadata("mcmc-runtime") or 0.5
        return int(float(run_time_hours) * 3600)

    @property
    def branch(self):
        """Branch to run the model with."""
        return buildkite.get_metadata("mcmc-branch") or "master"


class FullModelRunParams(BaseParams):
    """Params for a full model run"""

    BURN_IN_DEFAULT = 50

    @property
    def burn_in(self):
        if self.env_run_id:
            return self.BURN_IN_DEFAULT
        else:
            burn_in_option = buildkite.get_metadata("burn-in")
            return int(burn_in_option) if burn_in_option else self.BURN_IN_DEFAULT

    @property
    def use_latest_code(self):
        if self.env_run_id:
            return False
        else:
            use_latest_code = buildkite.get_metadata("use-latest-code") or "no"
            return use_latest_code == "yes"


class PowerBIParams(BaseParams):
    """Params for a powerbi post processing job"""
