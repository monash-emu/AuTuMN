"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import os
import click

from . import covid_19, marshall_islands, mongolia, sir_example

from .marshall_islands.calibration import run_calibration_chain as run_rmi_calibration_chain
from .mongolia.calibration import run_calibration_chain as run_mongolia_calibration_chain
from .covid_19.calibration.victoria import (
    run_vic_calibration_chain as run_victoria_covid_calibration_chain,
)
from .covid_19.calibration.malaysia import (
    run_mys_calibration_chain as run_malaysia_covid_calibration_chain,
)
from .covid_19.calibration.philippines import (
    run_phl_calibration_chain as run_philippines_covid_calibration_chain,
)
from autumn.db.models import create_power_bi_outputs, collate_outputs_powerbi
from autumn.plots.database_plots import plot_from_database


@click.group()
def cli():
    """AuTuMN CLI"""


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
        os.path.join(src_db_dir, fname) for fname in os.listdir(src_db_dir) if fname.endswith(".db")
    ]
    collate_outputs_powerbi(src_db_paths, dest_db_path, max_size_mb)


@click.group()
def run():
    """Run a model"""


@run.command("covid")
@click.argument("country", type=click.Choice(covid_19.COUNTRY_RUNNERS))
def run_covid(country):
    """Run the COVID model for some country"""
    runner = getattr(covid_19, country)
    runner.run_model()


@run.command("sir_example")
@click.argument("country", type=click.Choice(sir_example.COUNTRY_RUNNERS))
def run_sir_example(country):
    """Run the SIR model for some country"""
    runner = getattr(sir_example, country)
    runner.run_model()


@run.command("rmi")
def run_rmi():
    """Run the Marshall Islands TB model"""
    marshall_islands.run_model()


@run.command("mongolia")
def run_mongolia():
    """Run the Mongolia TB model"""
    mongolia.run_model()


@click.group()
def calibrate():
    """
    Calibrate a model
    """


@calibrate.command("rmi")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def rmi_calibration(max_seconds, run_id):
    """Run Marshall Islands model calibration."""
    marshall_islands.calibration.run_calibration_chain(max_seconds, run_id)


@calibrate.command("mongolia")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def mongolia_calibration(max_seconds, run_id):
    """Run Mongolia model calibration."""
    mongolia.calibration.run_calibration_chain(max_seconds, run_id)


@calibrate.command("victoria")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def victoria_calibration(max_seconds, run_id):
    """Run Victoria COVID model calibration."""
    run_victoria_covid_calibration_chain(max_seconds, run_id)


@calibrate.command("malaysia")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def malaysia_calibration(max_seconds, run_id):
    """Run Malaysia COVID model calibration."""
    run_malaysia_covid_calibration_chain(max_seconds, run_id)


@calibrate.command("philippines")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def philippines_calibration(max_seconds, run_id):
    """Run Malaysia COVID model calibration."""
    run_philippines_covid_calibration_chain(max_seconds, run_id)


cli.add_command(run)
cli.add_command(calibrate)
cli.add_command(db)
cli()
