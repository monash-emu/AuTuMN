"""
Runs AuTuMN applications

You can access this script from your CLI by running:

    python -m applications --help

"""
import click


from marshall_islands.runnners import run_rmi_model
from covid_19.runnners import run_covid_aus_model, run_covid_phl_model
from applications.mongolia.mongolia_calibration import (
    run_calibration_chain as run_mongolia_calibration_chain,
)
from applications.marshall_islands.rmi_calibration import (
    run_calibration_chain as run_rmi_calibration_chain,
)
from applications.covid_19.covid_calibration import (
    run_calibration_chain as run_covid_calibration_chain,
)


@click.group()
def cli():
    """
    AuTuMN utility command line
    """
    pass


@click.command()
def rmi():
    """
    Run the Marshall Islands model.
    """
    rmi_run_model()


@click.command()
def covidaus():
    """
    Run the COVID Australian model.
    """
    run_covid_aus_model()


@click.command()
def covidphl():
    """
    Run the COVID Phillipines model.
    """
    run_covid_phl_model()


@click.command()
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def rmi_calibration(max_seconds, run_id):
    """
    Run RMI model calibration.
    """
    run_rmi_calibration_chain(max_seconds, run_id)


@click.command()
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def mongolia_calibration(max_seconds, run_id):
    """
    Run Mongolia model calibration.
    """
    run_mongolia_calibration_chain(max_seconds, run_id)


@click.command()
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
def covid_calibration(max_seconds, run_id, country):
    """
    Run Covid model calibration.
    """
    run_covid_calibration_chain(max_seconds, run_id, country)


cli.add_command(rmi_calibration)
cli.add_command(mongolia_calibration)
cli.add_command(covid_calibration)
cli.add_command(rmi)
cli.add_command(covidaus)
cli.add_command(covidphl)
cli()
