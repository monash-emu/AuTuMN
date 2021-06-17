"""
Runs remote tasks

You can access this script from your CLI by running:

    python -m tasks --help

"""
import os
import warnings
import logging

import click


@click.group()
def tasks():
    """
    Run remote task pipelines.
    """


@tasks.command("calibrate")
@click.option("--run", type=str, required=True)
@click.option("--chains", type=int, required=True)
@click.option("--runtime", type=int, required=True)
@click.option("--verbose", is_flag=True)
def run_calibrate(run, chains, runtime, verbose):
    pre_task_setup()

    from autumn.tasks.calibrate import calibrate_task

    calibrate_task(run, runtime, chains, verbose)


@tasks.command("full")
@click.option("--run", type=str, required=True)
@click.option("--burn", type=int, required=True)
@click.option("--sample", type=int, required=True)
@click.option("--verbose", is_flag=True)
def run_full_models(run, burn, sample, verbose):
    pre_task_setup()

    from autumn.tasks.full import full_model_run_task

    full_model_run_task(run, burn, sample, not verbose)


@tasks.command("powerbi")
@click.option("--run", type=str, required=True)
@click.option("--urunid", type=str, default="mle")
@click.option("--verbose", is_flag=True)
def run_powerbi(run, urunid, verbose):
    pre_task_setup()

    from autumn.tasks.powerbi import powerbi_task

    powerbi_task(run, urunid, not verbose)


def pre_task_setup():
    setup_warnings()
    #setup_sentry()
    setup_matplotlib()
    setup_logging()


def setup_sentry():
    """
    Setup Sentry error reporting
    https://sentry.io/organizations/monash-emu/issues/?project=5807435
    """
    import sentry_sdk

    SENTRY_DSN = os.environ.get("SENTRY_DSN")
    if SENTRY_DSN:
        sentry_sdk.init(SENTRY_DSN, in_app_include=["autumn", "summer"])


def setup_matplotlib():
    import matplotlib

    # Use non-TK matplotlib backend to avoid issues with pseudo-windows or something.
    # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread/29172195#29172195
    matplotlib.use("Agg")


def setup_warnings():
    # Ignore noisy deprecation warnings.
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)


def setup_logging():

    from autumn.settings import LOGGING_DIR, REMOTE_BASE_DIR

    # Configure logging for the Boto3 library
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("nose").setLevel(logging.WARNING)

    # Side effects yay!
    os.makedirs(REMOTE_BASE_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    from autumn.tasks.utils import set_logging_config

    set_logging_config(verbose=True)
