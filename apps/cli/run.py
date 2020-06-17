"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from apps import covid_19, marshall_islands, mongolia, sir_example


@click.group()
def run():
    """Run a model"""


@run.command("covid")
@click.argument("region", type=click.Choice(covid_19.REGION_APPS))
def run_covid(region):
    """Run the COVID model for some region"""
    region_app = covid_19.get_region_app(region)
    region_app.run_model()


@run.command("sir_example")
@click.argument("region", type=click.Choice(sir_example.REGION_APPS))
def run_sir_example(region):
    """Run the SIR model for some region"""
    region_app = sir_example.get_region_app(region)
    region_app.run_model()


@run.command("rmi")
def run_rmi():
    """Run the Marshall Islands TB model"""
    marshall_islands.run_model()


@run.command("mongolia")
def run_mongolia():
    """Run the Mongolia TB model"""
    mongolia.run_model()
