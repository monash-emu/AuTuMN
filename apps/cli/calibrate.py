"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from apps.marshall_islands.calibration import (
    run_calibration_chain as run_rmi_calibration_chain,
)
from apps.mongolia.calibration import (
    run_calibration_chain as run_mongolia_calibration_chain,
)
from apps.covid_19.calibration.victoria import (
    run_vic_calibration_chain as run_victoria_covid_calibration_chain,
)
from apps.covid_19.calibration.base import (
    run_full_models_for_mcmc as run_full_covid_models_for_mcmc,
)
from apps.covid_19.calibration.malaysia import (
    run_mys_calibration_chain as run_malaysia_covid_calibration_chain,
)
from apps.covid_19.calibration.philippines import (
    run_phl_calibration_chain as run_philippines_covid_calibration_chain,
)


@click.group()
def calibrate():
    """
    Calibrate a model
    """


@calibrate.command("rmi")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def rmi_calibration(max_seconds, run_id):
    """Run Marshall Islands model calibration."""
    marshall_islands.calibration.run_calibration_chain(max_seconds, run_id)


@calibrate.command("mongolia")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def mongolia_calibration(max_seconds, run_id):
    """Run Mongolia model calibration."""
    mongolia.calibration.run_calibration_chain(max_seconds, run_id)


@calibrate.command("victoria")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def victoria_calibration(max_seconds, run_id):
    """Run Victoria COVID model calibration."""
    run_victoria_covid_calibration_chain(max_seconds, run_id)


@calibrate.command("malaysia")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def malaysia_calibration(max_seconds, run_id):
    """Run Malaysia COVID model calibration."""
    run_malaysia_covid_calibration_chain(max_seconds, run_id)


@calibrate.command("philippines")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def philippines_calibration(max_seconds, run_id):
    """Run Malaysia COVID model calibration."""
    run_philippines_covid_calibration_chain(max_seconds, run_id)

