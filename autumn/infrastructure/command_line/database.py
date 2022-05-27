"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m autumn db --help

"""
import click


@click.group()
def db():
    """Database utilities"""


@db.command("fetch")
def download_input_data():
    """
    Fetch input data from external sources for input database.
    """
    from autumn.tools.inputs import fetch_input_data

    fetch_input_data()


@db.command("build")
def build_input_db():
    """
    Build a new input database from input data files.
    """
    from autumn.tools.inputs import build_input_database

    build_input_database(rebuild=True)


@db.command("feather2sql")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def feather2sql(src_db_path, dest_db_path):
    """
    Convert a Feather DB to a SQLite DB
    """
    from autumn.tools import db as autumn_db

    assert autumn_db.FeatherDatabase.is_compatible(
        src_db_path
    ), "Source DB must be FeatherDatabase compatible"
    src_db = autumn_db.FeatherDatabase(src_db_path)
    autumn_db.database.convert_database(src_db, autumn_db.database.Database, dest_db_path)


@db.command("parquet2sql")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def parquet2sql(src_db_path, dest_db_path):
    """
    Convert a Feather DB to a SQLite DB
    """
    from autumn.tools import db as autumn_db

    assert autumn_db.ParquetDatabase.is_compatible(
        src_db_path
    ), "Source DB must be ParquetDatabase compatible"
    src_db = autumn_db.ParquetDatabase(src_db_path)
    autumn_db.database.convert_database(src_db, autumn_db.database.Database, dest_db_path)
