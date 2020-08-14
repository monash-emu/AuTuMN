"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from apps import covid_19, sir_example


@click.group()
def calibrate():
    """
    Calibrate a model
    """


for region in covid_19.app.region_names:

    @calibrate.command(region)
    @click.argument("max_seconds", type=int)
    @click.argument("run_id", type=int)
    @click.option("--num-chains", type=int, default=1)
    def run_region_calibration(max_seconds, run_id, num_chains, region=region):
        """Run COVID model calibration for region"""
        covid_region = covid_19.app.get_region(region)
        covid_region.calibrate_model(max_seconds, run_id, num_chains)


@calibrate.group()
def example():
    """Calibrate the sir_example model"""


for region in sir_example.app.region_names:

    @example.command(region)
    @click.argument("max_seconds", type=int)
    @click.argument("run_id", type=int)
    @click.option("--num-chains", type=int, default=1)
    def run_region_calibration(max_seconds, run_id, num_chains, region=region):
        """Run COVID model calibration for region"""
        sir_region = sir_example.app.get_region(region)
        sir_region.calibrate_model(max_seconds, run_id, num_chains)
