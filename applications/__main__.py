"""
Runs AuTuMN applications

You can access this script from your CLI by running:

    python -m applications --help

"""
import click

from .marshall_islands.rmi_experiment import run_model as rmi_run_model
from .vietnam.model import run_model as run_vietnam_model


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


cli.add_command(vietnam)
cli.add_command(rmi)
cli()
