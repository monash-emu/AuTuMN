"""
Runs AuTuMN applications

You can access this script from your CLI by running:

    python -m applications --help

"""
import click

from . import covid_19, marshall_islands, mongolia

from .marshall_islands.calibration import run_calibration_chain as run_rmi_calibration_chain
from .mongolia.calibration import run_calibration_chain as run_mongolia_calibration_chain


@click.group()
def cli():
    """AuTuMN CLI"""


@click.group()
def run():
    """Run a model"""


@run.command("covid-aus")
def run_covid_aus():
    """Run the COVID Australia model"""
    covid_19.aus.run_model()


@run.command("covid-phl")
def run_covid_phl():
    """Run the COVID Phillipines model"""
    covid_19.phl.run_model()


@run.command("rmi")
def run_rmi():
    """Run the Marshall Islands TB model"""
    marshall_islands.run_model()


@run.command("mongolia")
def run_mongolia():
    """Run the Mongolia TB model"""
    mongolia.run_model()


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
    run_rmi_calibration_chain(max_seconds, run_id)


@calibrate.command("mongolia")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def mongolia_calibration(max_seconds, run_id):
    """Run Mongolia model calibration."""
    run_mongolia_calibration_chain(max_seconds, run_id)


cli.add_command(run)
cli.add_command(calibrate)
cli()
