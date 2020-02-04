"""
Runs AuTuMN applications

You can access this script from your CLI by running:

    python -m applications --help

"""
import click

from .marshall_islands.marshall_islands import run_model as rmi_run_model

@click.group()
def cli():
    """
    AuTuMN utility command line
    """
    pass


@click.command()
def rmi():
    """
    Run the Marshall Islands model.
    """
    rmi_run_model()


cli.add_command(rmi)
cli()
