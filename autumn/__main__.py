"""
Runs module utility scripts
You can access these script from your CLI by running:

    python -m autumn --help

"""
import click

from .db.input_data import build_input_database

@click.group()
def cli():
    """
    AuTuMN utility command line
    """
    pass


@click.command()
def build_input_db():
    """
    Build a new, timestamped input database from Excel files.
    """
    build_input_database()


cli.add_command(build_input_db)
cli()
