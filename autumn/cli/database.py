"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m autumn db --help

"""
import click

from autumn import db as autumn_db
from autumn.inputs import build_input_database, fetch_input_data


@click.group()
def db():
    """Database utilities"""


@db.command("fetch")
def download_input_data():
    """
    Fetch input data from external sources for input database.
    """
    fetch_input_data()


@db.command("build")
@click.option("--force", is_flag=True)
def build_input_db(force):
    """
    Build a new input database from input data files.
    """
    build_input_database(force)


@db.command("feather2sql")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def feather2sql(src_db_path, dest_db_path):
    """
    Convert a Feather DB to a SQLite DB
    """
    assert autumn_db.FeatherDatabase.is_compatible(
        src_db_path
    ), "Source DB must be FeatherDatabase compatible"
    src_db = autumn_db.FeatherDatabase(src_db_path)
    autumn_db.database.convert_database(src_db, autumn_db.database.Database, dest_db_path)
