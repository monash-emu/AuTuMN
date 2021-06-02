"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from apps import covid_19, sir_example, tuberculosis, tuberculosis_strains


@click.group()
def calibrate():
    """
    Calibrate a model
    """


@calibrate.command("tbs")
@click.argument("region", type=click.Choice(tuberculosis_strains.app.region_names))
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
@click.option("--num-chains", type=int, default=1)
def run_tb_calibration(region, max_seconds, run_id, num_chains):
    """Run tuberculosis model calibration"""
    app_region = tuberculosis_strains.app.get_region(region)
    app_region.calibrate_model(max_seconds, run_id, num_chains)


@calibrate.command("tb")
@click.argument("region", type=click.Choice(tuberculosis.app.region_names))
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
@click.option("--num-chains", type=int, default=1)
def run_tb_calibration(region, max_seconds, run_id, num_chains):
    """Run tuberculosis model calibration"""
    app_region = tuberculosis.app.get_region(region)
    app_region.calibrate_model(max_seconds, run_id, num_chains)


@calibrate.command("covid")
@click.argument("region", type=click.Choice(covid_19.app.region_names))
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
@click.option("--num-chains", type=int, default=1)
def run_covid_calibration(region, max_seconds, run_id, num_chains):
    """Run COVID model calibration"""
    app_region = covid_19.app.get_region(region)
    app_region.calibrate_model(max_seconds, run_id, num_chains)


@calibrate.command("example")
@click.argument("region", type=click.Choice(sir_example.app.region_names))
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
@click.option("--num-chains", type=int, default=1)
def run_example_calibration(region, max_seconds, run_id, num_chains):
    """Run example model calibration"""
    app_region = sir_example.app.get_region(region)
    app_region.calibrate_model(max_seconds, run_id, num_chains)
