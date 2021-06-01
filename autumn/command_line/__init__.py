"""
Runs Autumn utilities

You can access this script from your CLI by running:

    python -m autumn --help

"""
import click

from .database import db
from .secrets import secrets
from .projects import project
from .remote import remote
from .tasks import tasks


@click.group()
def cli():
    """Autumn project command line"""


cli.add_command(project)
cli.add_command(db)
cli.add_command(secrets)
cli.add_command(remote)
cli.add_command(tasks)
