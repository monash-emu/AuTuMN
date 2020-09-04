"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m autumn db --help

"""
import os
import click

from autumn.db import models
from autumn.inputs import build_input_database, fetch_input_data
from autumn.plots.database_plots import plot_from_mcmc_databases
from autumn.plots.uncertainty_plots import plot_timeseries_with_uncertainty


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


@db.command("plot-mcmc")
@click.argument("app_name", type=str)
@click.argument("region_name", type=str)
@click.argument("src_db_dir", type=str)
@click.argument("plot_dir", type=str)
def plot_mcmc_database(app_name, region_name, src_db_dir, plot_dir):
    """Plot data from a MCMC run folder"""
    assert os.path.isdir(src_db_dir), f"{src_db_dir} must be a folder"
    assert os.path.isdir(plot_dir), f"{plot_dir} must be a folder"
    plot_from_mcmc_databases(app_name, region_name, src_db_dir, plot_dir)


@db.command("plot-uncertainty")
@click.argument("region", type=str)
@click.argument("src_db_dir", type=str)
@click.argument("plot_dir", type=str)
def plot_mcmc_database(region, src_db_dir, plot_dir):
    """
    Plot data from a MCMC database with uncertainty weights to a plot folder
    Assumes a COVID model.
    """
    assert os.path.isfile(src_db_dir), f"{src_db_dir} must be a file"
    assert os.path.isdir(plot_dir), f"{plot_dir} must be a folder"
    plot_timeseries_with_uncertainty(region, src_db_dir, plot_dir)


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
    models.collate_databases(src_db_paths, dest_db_path)


@db.command("prune")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def prune(src_db_path, dest_db_path):
    """
    Drop data for all outputs except for the MLE model run.
    Requires MCMC run metadata.
    """
    assert os.path.isfile(src_db_path), f"{src_db_path} must be a file"
    models.prune(src_db_path, dest_db_path)


@db.command("unpivot")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def unpivot(src_db_path, dest_db_path):
    """
    Convert model outputs into PowerBI-friendly unpivoted format.
    """
    assert os.path.isfile(src_db_path), f"{src_db_path} must be a file"
    models.unpivot(src_db_path, dest_db_path)

