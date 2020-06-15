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
