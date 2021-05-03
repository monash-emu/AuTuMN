"""
Runs Autumn utilities

You can access this script from your CLI by running:

    python -m autumn --help

"""
import click

from .database import db
from .secrets import secrets


@click.group()
def cli():
    """Autumn Utilities CLI"""


cli.add_command(db)
cli.add_command(secrets)
