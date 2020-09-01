"""
Runs Autumn apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from .run import run
from .calibrate import calibrate


@click.group()
def cli():
    """Autumn Apps CLI"""


cli.add_command(run)
cli.add_command(calibrate)
cli()
