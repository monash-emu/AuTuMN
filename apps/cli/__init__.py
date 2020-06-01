"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from .database import db
from .run import run
from .run_mcmc import run_mcmc
from .calibrate import calibrate


@click.group()
def cli():
    """AuTuMN CLI"""


cli.add_command(run)
cli.add_command(run_mcmc)
cli.add_command(calibrate)
cli.add_command(db)
cli()
