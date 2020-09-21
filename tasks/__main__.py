"""
Runs remote tasks

You can access this script from your CLI by running:

    python -m tasks --help

"""
import os
import sys
import logging
import warnings

import click
import sentry_sdk

# Ignore noisy deprecation warnings.
warnings.simplefilter(action="ignore", category=FutureWarning)

# Setup Sentry error reporting - https://sentry.io/welcome/
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(SENTRY_DSN)

# Configure command logging
logging.basicConfig(format="%(asctime)s %(module)s:%(levelname)s: %(message)s", level=logging.INFO)

# Configure logging for the Boto3 library
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("nose").setLevel(logging.WARNING)

# Configure Luigi config.
os.environ["LUIGI_CONFIG_PATH"] = "tasks/config/luigi.cfg"

import luigi

from .settings import BASE_DIR
from .calibrate import RunCalibrate
from .full_model_run import RunFullModels
from .powerbi import RunPowerBI
from .dhhs import RunDHHS


logger = logging.getLogger(__name__)

os.makedirs(BASE_DIR, exist_ok=True)


@click.group()
def cli():
    """
    Run Luigi pipelines.
    """


@cli.command("calibrate")
@click.option("--run", type=str, required=True)
@click.option("--chains", type=int, required=True)
@click.option("--runtime", type=int, required=True)
@click.option("--workers", type=int, required=True)
def run_calibrate(run, chains, runtime, workers):
    """
    Run calibration pipeline.
    """
    task = RunCalibrate(run_id=run, num_chains=chains, runtime=runtime)
    result = luigi.build([task], workers=workers, local_scheduler=True, detailed_summary=True)
    _handle_result(result)


@cli.command("full")
@click.option("--run", type=str, required=True)
@click.option("--burn", type=int, required=True)
@click.option("--workers", type=int, required=True)
def run_full_models(run, burn, workers):
    """
    Run full model pipeline.
    """
    task = RunFullModels(run_id=run, burn_in=burn)
    result = luigi.build([task], workers=workers, local_scheduler=True, detailed_summary=True)
    _handle_result(result)


@cli.command("powerbi")
@click.option("--run", type=str, required=True)
@click.option("--workers", type=int, required=True)
def run_powerbi(run, workers):
    """
    Run PowerBI post-processing.
    """
    task = RunPowerBI(run_id=run)
    result = luigi.build([task], workers=workers, local_scheduler=True, detailed_summary=True)
    _handle_result(result)


@cli.command("dhhs")
@click.option("--commit", type=str, required=True)
@click.option("--workers", type=int, required=True)
def run_dhhs(commit, workers):
    """
    Run DHHS post processing.
    """
    task = RunDHHS(commit=commit)
    result = luigi.build([task], workers=workers, local_scheduler=True, detailed_summary=True)
    _handle_result(result)


def _handle_result(result: luigi.LuigiStatusCode):
    if not result.scheduling_succeeded:
        emoji, explanation = result.status.value
        logger.info("Luigi task failed %s %s", emoji, explanation)
        sys.exit(-1)


cli()
