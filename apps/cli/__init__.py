"""
Runs Autumn apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from apps.tuberculosis.regions.marshall_islands.outputs.main_script import (
    make_all_rmi_plots,
)

from apps.covid_19.mixing_optimisation.sample_code import run_sample_code

from .calibrate import calibrate
from .run import run


@click.group()
def cli():
    """Autumn Apps CLI"""


@cli.command("plotrmi")
def plotting_cli():
    """Plot all model outputs for the Marshall Islands project"""
    make_all_rmi_plots()


@cli.command("runsampleopti")
def run_sample():
    """Run sample code for optimisation"""
    run_sample_code()


cli.add_command(run)
cli.add_command(calibrate)
