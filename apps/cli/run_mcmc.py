"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from apps import covid_19
from apps.covid_19.calibration import run_full_models_for_mcmc


@click.group()
def run_mcmc():
    """
    Run all accepted models from a MCMC chain.
    """


for region in covid_19.app.region_names:

    @run_mcmc.command(region)
    @click.argument("burn_in", type=int)
    @click.argument("src_db_path", type=str)
    @click.argument("dest_db_path", type=str)
    def run_mcmc_func(burn_in, src_db_path, dest_db_path, region=region):
        run_full_models_for_mcmc(region, burn_in, src_db_path, dest_db_path)
