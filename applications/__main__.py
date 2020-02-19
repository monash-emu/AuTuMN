"""
Runs AuTuMN applications

You can access this script from your CLI by running:

    python -m applications --help

"""
import click

from applications.marshall_islands.rmi_experiment import run_model as rmi_run_model
from applications.vietnam.model import run_model as run_vietnam_model
from applications.mongolia.mongolia_calibration import (
    run_calibration_chain as run_mongolia_calibration_chain,
)
from applications.marshall_islands.rmi_calibration import (
    run_calibration_chain as run_rmi_calibration_chain,
)


@click.group()
def cli():
    """
    AuTuMN utility command line
    """
    pass


@click.command()
def vietnam():
    """
    Run the Marshall Islands model.
    """
    run_vietnam_model()


@click.command()
def rmi():
    """
    Run the Marshall Islands model.
    """
    rmi_run_model()


@click.command()
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def rmi_calibration(max_seconds, run_id):
    """
    Run RMI model calibration.
    """
    run_rmi_calibration_chain(max_seconds, run_id)


@click.command()
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def mongolia_calibration(max_seconds, run_id):
    """
    Run Mongolia model calibration.
    """
    run_mongolia_calibration_chain(max_seconds, run_id)


cli.add_command(rmi_calibration)
cli.add_command(mongolia_calibration)
cli.add_command(vietnam)
cli.add_command(rmi)
cli()
