"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from apps.covid_19.calibration.base import (
    run_full_models_for_mcmc as run_full_covid_models_for_mcmc,
)


@click.group()
def run_mcmc():
    """
    Run a model based on the parameters from a a MCMC chain.
    """


@run_mcmc.command("malaysia")
@click.argument("burn_in", type=int)
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def run_mcmc_malaysia(burn_in, src_db_path, dest_db_path):
    """
    Run the Malaysia COVID model based on the parameters from a a MCMC chain.
    """
    run_full_covid_models_for_mcmc("malaysia", burn_in, src_db_path, dest_db_path)


@run_mcmc.command("philippines")
@click.argument("burn_in", type=int)
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def run_mcmc_philippines(burn_in, src_db_path, dest_db_path):
    """
    Run the Philippines COVID model based on the parameters from a a MCMC chain.
    """
    run_full_covid_models_for_mcmc("philippines", burn_in, src_db_path, dest_db_path)


@run_mcmc.command("victoria")
@click.argument("burn_in", type=int)
@click.argument("src_db_path", type=str)
@click.argument("dest_db_path", type=str)
def run_mcmc_victoria(burn_in, src_db_path, dest_db_path):
    """
    Run the Victoria COVID model based on the parameters from a a MCMC chain.
    """
    run_full_covid_models_for_mcmc("victoria", burn_in, src_db_path, dest_db_path)
