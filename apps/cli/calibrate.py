"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from apps.covid_19 import calibration as covid_calibration
from apps.marshall_islands import calibration as rmi_calibration
from apps.mongolia import calibration as mongolia_calibration


@click.group()
def calibrate():
    """
    Calibrate a model
    """


for region in covid_calibration.CALIBRATIONS.keys():

    @calibrate.command(region)
    @click.argument("max_seconds", type=int)
    @click.argument("run_id", type=int)
    def run_region_calibration(max_seconds, run_id, region=region):
        """Run COVID model calibration for region"""
        calib_func = covid_calibration.get_calibration_func(region)
        calib_func(max_seconds, run_id)


@calibrate.command("mongolia")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def mongolia_calibration(max_seconds, run_id):
    """Run Mongolia TB model calibration."""
    mongolia_calibration.run_calibration_chain(max_seconds, run_id)


@calibrate.command("rmi")
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def rmi_calibration(max_seconds, run_id):
    """Run Marshall Islands TB model calibration."""
    rmi_calibration.run_calibration_chain(max_seconds, run_id)
