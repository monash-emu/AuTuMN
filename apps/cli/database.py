"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import os
import click

from autumn.db.models import create_power_bi_outputs, collate_outputs_powerbi
from autumn.plots.database_plots import plot_from_database


@click.group()
def db():
    """Database utilities"""


@db.command("plot")
@click.argument("model_run_path", type=str)
def plot_database(model_run_path):
    """Re-plot data from a model run folder"""
    plot_from_database(model_run_path)


@db.command("powerbi")
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def powerbi_convert(src_db_path, dest_db_path):
    """Convert model outputs into PowerBI format"""
    assert os.path.isfile(src_db_path), f"{src_db_path} must be a file"
    create_power_bi_outputs(src_db_path, dest_db_path)


@db.command("powerbi-collate")
@click.argument("src_db_dir", type=str)
@click.argument("dest_db_path", type=str)
@click.argument("max_size_mb", type=int)
def powerbi_collate(src_db_dir, dest_db_path, max_size_mb):
    """Collate MCMC databases and then convert model outputs into PowerBI format"""
    assert os.path.isdir(src_db_dir), f"{src_db_dir} must be a folder"
    src_db_paths = [
        os.path.join(src_db_dir, fname)
        for fname in os.listdir(src_db_dir)
        if fname.endswith(".db")
    ]
    collate_outputs_powerbi(src_db_paths, dest_db_path, max_size_mb)

