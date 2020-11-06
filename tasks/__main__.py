"""
Runs remote tasks

You can access this script from your CLI by running:

    python -m tasks --help

"""
import os
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

from .settings import BASE_DIR
from .calibrate import calibrate_task
from .full import full_model_run_task
from .powerbi import powerbi_task
from .dhhs import dhhs_task


logger = logging.getLogger(__name__)

os.makedirs(BASE_DIR, exist_ok=True)


@click.group()
def cli():
    """
    Run remote task pipelines.
    """


@cli.command("calibrate")
@click.option("--run", type=str, required=True)
@click.option("--chains", type=int, required=True)
@click.option("--runtime", type=int, required=True)
@click.option("--workers", type=int, required=False)  # Backwards compatibility.
@click.option("--verbose", is_flag=True)
def run_calibrate(run, chains, runtime, workers, verbose):
    calibrate_task(run, runtime, chains, quiet=not verbose)


@cli.command("full")
@click.option("--run", type=str, required=True)
@click.option("--burn", type=int, required=True)
@click.option("--workers", type=int, required=False)  # Backwards compatibility.
@click.option("--verbose", is_flag=True)
def run_full_models(run, burn, workers, verbose):
    full_model_run_task(run, burn, quiet=not verbose)


@cli.command("powerbi")
@click.option("--run", type=str, required=True)
@click.option("--workers", type=int, required=False)  # Backwards compatibility.
@click.option("--verbose", is_flag=True)
def run_powerbi(run, workers, verbose):
    powerbi_task(run, quiet=not verbose)


@cli.command("dhhs")
@click.option("--commit", type=str, required=True)
@click.option("--workers", type=int, required=False)  # Backwards compatibility.
@click.option("--verbose", is_flag=True)
def run_dhhs(commit, workers, verbose):
    dhhs_task(commit, quiet=not verbose)


cli()
