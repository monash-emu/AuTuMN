"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import os
import click

from autumn.db import models
from autumn.plots.database_plots import plot_from_database, plot_from_mcmc_databases
from autumn.tool_kit.uncertainty import (
    add_uncertainty_weights,
    add_uncertainty_quantiles,
)


@click.group()
def db():
    """Database utilities"""


@db.command("plot")
@click.argument("model_run_path", type=str)
def plot_database(model_run_path):
    """Re-plot data from a model run folder"""
    plot_from_database(model_run_path)


@db.command("plot-mcmc")
@click.argument("src_db_dir", type=str)
@click.argument("plot_dir", type=str)
def plot_mcmc_database(src_db_dir, plot_dir):
    """Plot data from a MCMC run folder"""
    assert os.path.isdir(src_db_dir), f"{src_db_dir} must be a folder"
    assert os.path.isdir(plot_dir), f"{plot_dir} must be a folder"
    plot_from_mcmc_databases(src_db_dir, plot_dir)


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


@click.group()
def uncertainty():
    """Calculate uncertainty around MCMC calibrated parameters"""


@uncertainty.command("weights")
@click.argument("output_name", type=str)
@click.argument("db_path", type=str)
def uncertainty_weights(output_name: str, db_path: str):
    """
    Calculate uncertainty weights for the specified derived outputs.
    Requires MCMC run metadata.
    """
    assert os.path.isfile(db_path), f"{db_path} must be a file"
    add_uncertainty_weights(output_name, db_path)


@uncertainty.command("quantiles")
@click.argument("db_path", type=str)
def uncertainty_quantiles(db_path: str):
    """
    Add uncertainty quantiles for the any derived outputs with weights.
    Requires MCMC run metadata abd .
    """
    assert os.path.isfile(db_path), f"{db_path} must be a file"
    add_uncertainty_quantiles(db_path)


db.add_command(uncertainty)
