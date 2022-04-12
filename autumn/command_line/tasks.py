"""
Runs remote tasks

You can access this script from your CLI by running:

    python -m tasks --help

"""
import os
import warnings
import logging
from importlib import import_module

import click
import json

from autumn.tools.runs.managed import LocalRunStore


@click.group()
def tasks():
    """
    Run remote task pipelines.
    """

@tasks.command("generic")
@click.argument("task_spec", type=str)
def run_generic(task_spec: str):
    ts_dict = json.loads(task_spec)
    m = import_module(ts_dict['task_module'])
    f = getattr(m, ts_dict['task_func'])
    f(**ts_dict['task_args'])

@tasks.command("calibrate")
@click.option("--run", type=str, required=True)
@click.option("--chains", type=int, required=True)
@click.option("--runtime", type=int, required=True)
@click.option("--verbose", is_flag=True)
def run_calibrate(run, chains, runtime, verbose):
    pre_task_setup(run)

    from autumn.tasks.calibrate import calibrate_task

    calibrate_task(run, runtime, chains, verbose)

@tasks.command("resume_calibration")
@click.option("--run", type=str, required=True)
@click.option("--baserun", type=str, required=True)
@click.option("--runtime", type=int, required=True)
@click.option("--chains", type=int, required=True)
@click.option("--verbose", is_flag=True)
def resume_calibration(run, baserun, runtime, chains, verbose):
    pre_task_setup(run)

    from autumn.tasks.resume import resume_calibration_task

    resume_calibration_task(run, baserun, runtime, chains, verbose)


@tasks.command("full")
@click.option("--run", type=str, required=True)
@click.option("--burn", type=int, required=True)
@click.option("--sample", type=int, required=True)
@click.option("--verbose", is_flag=True)
def run_full_models(run, burn, sample, verbose):
    pre_task_setup(run)

    from autumn.tasks.full import full_model_run_task

    full_model_run_task(run, burn, sample, not verbose)


@tasks.command("powerbi")
@click.option("--run", type=str, required=True)
@click.option("--urunid", type=str, default="mle")
@click.option("--verbose", is_flag=True)
def run_powerbi(run, urunid, verbose):
    pre_task_setup(run)

    from autumn.tasks.powerbi import powerbi_task

    powerbi_task(run, urunid, not verbose)


def pre_task_setup(run_id: str):
    setup_warnings()
    setup_matplotlib()
    setup_logging(run_id)


def setup_matplotlib():
    import matplotlib

    # Use non-TK matplotlib backend to avoid issues with pseudo-windows or something.
    # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread/29172195#29172195
    matplotlib.use("Agg")


def setup_warnings():
    # Ignore noisy deprecation warnings.
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)


def setup_logging(run_id: str):

    # Configure logging for the Boto3 library
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("nose").setLevel(logging.WARNING)

    lrs = LocalRunStore()

    mr = lrs.create(run_id)

    from autumn.tasks.utils import set_logging_config

    set_logging_config(verbose=True, log_path = mr.local_path / 'logs')
