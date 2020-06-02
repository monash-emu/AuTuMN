"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import os
import click

from autumn.db.models import collate_databases
from autumn.plots.database_plots import plot_from_database


@click.group()
def db():
    """Database utilities"""


@db.command("plot")
@click.argument("model_run_path", type=str)
def plot_database(model_run_path):
    """Re-plot data from a model run folder"""
    plot_from_database(model_run_path)


@db.command("collate")
@click.argument("src_db_dir", type=str)
@click.argument("dest_db_path", type=str)
def collate(src_db_dir, dest_db_path):
    """
    Merge all databases from a folder into a single database.
    """
    assert os.path.isdir(src_db_dir), f"{src_db_dir} must be a folder"
    src_db_paths = [
        os.path.join(src_db_dir, fname)
        for fname in os.listdir(src_db_dir)
        if fname.endswith(".db")
    ]
    collate_databases(src_db_paths, dest_db_path)


@db.command("uncertainty")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def uncertainty(src_db_path, dest_db_path):
    """
    Add uncertainty estimates to specified derived outputs.
    Requires MCMC run metadata.
    FIXME: Do we need to specify derived outputs?
    """
    assert os.path.isfile(src_db_path), f"{src_db_path} must be a file"
    xxxxxxxxxxxxxx(src_db_path, dest_db_path)


@db.command("prune")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def prune(src_db_path, dest_db_path):
    """
    Drop data for all outputs except for the MLE model run.
    Requires MCMC run metadata.
    """
    assert os.path.isfile(src_db_path), f"{src_db_path} must be a file"
    xxxxxxxxxxxxxx(src_db_path, dest_db_path)


@db.command("unpivot")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def unpivot(src_db_path, dest_db_path):
    """
    Convert model outputs into PowerBI-friendly unpivoted format.
    """
    assert os.path.isfile(src_db_path), f"{src_db_path} must be a file"
    xxxxxxxxxxxxxx(src_db_path, dest_db_path)
