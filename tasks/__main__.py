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
warnings.simplefilter(action="ignore", category=UserWarning)

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
from .calibrate import calibrate_task
from .full import full_model_run_task
from .powerbi import powerbi_task
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
@click.option("--workers", type=int, required=False)  # Backwards compat.
@click.option("--verbose", is_flag=True)
def run_calibrate(run, chains, runtime, workers, verbose):
    calibrate_task(run, runtime, chains, quiet=not verbose)


@cli.command("full")
@click.option("--run", type=str, required=True)
@click.option("--burn", type=int, required=True)
@click.option("--workers", type=int, required=False)  # Backwards compat.
@click.option("--verbose", is_flag=True)
def run_full_models(run, burn, workers, verbose):
    full_model_run_task(run, burn, quiet=not verbose)


@cli.command("powerbi")
@click.option("--run", type=str, required=True)
@click.option("--workers", type=int, required=False)  # Backwards compat.
@click.option("--verbose", is_flag=True)
def run_powerbi(run, workers, verbose):
    powerbi_task(run, quiet=not verbose)


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
