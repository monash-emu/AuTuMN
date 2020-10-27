"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m autumn db --help

"""
import os
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


@db.command("collate")
@click.argument("src_db_dir", type=str)
@click.argument("dest_db_path", type=str)
def collate(src_db_dir, dest_db_path):
    """
    Merge all databases from a folder into a single database.
    """
    assert os.path.isdir(src_db_dir), f"{src_db_dir} must be a folder"
    src_db_paths = [
        os.path.join(src_db_dir, fname) for fname in os.listdir(src_db_dir) if fname.endswith(".db")
    ]
    autumn_db.process.collate_databases(src_db_paths, dest_db_path)


@db.command("prune")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def prune(src_db_path, dest_db_path):
    """
    Drop data for all outputs except for the MLE model run.
    Requires MCMC run metadata.
    """
    assert os.path.isfile(src_db_path), f"{src_db_path} must be a file"
    autumn_db.process.prune(src_db_path, dest_db_path)


@db.command("unpivot")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def unpivot(src_db_path, dest_db_path):
    """
    Convert model outputs into PowerBI-friendly unpivoted format.
    """
    assert os.path.isfile(src_db_path), f"{src_db_path} must be a file"
    autumn_db.process.unpivot(src_db_path, dest_db_path)
