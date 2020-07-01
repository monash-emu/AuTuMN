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
@click.option("--no-scenarios", is_flag=True)
def run_covid(region, no_scenarios):
    """Run the COVID model for some region"""
    region_app = covid_19.get_region_app(region)
    region_app.run_model(run_scenarios=not no_scenarios)


@run.command("sir_example")
@click.argument("region", type=click.Choice(sir_example.REGION_APPS))
@click.option("--no-scenarios", is_flag=True)
def run_sir_example(region, no_scenarios):
    """Run the SIR model for some region"""
    region_app = sir_example.get_region_app(region)
    region_app.run_model(run_scenarios=not no_scenarios)


@run.command("rmi")
@click.option("--no-scenarios", is_flag=True)
def run_rmi(no_scenarios):
    """Run the Marshall Islands TB model"""
    marshall_islands.run_model(run_scenarios=not no_scenarios)


@run.command("mongolia")
@click.option("--no-scenarios", is_flag=True)
def run_mongolia(no_scenarios):
    """Run the Mongolia TB model"""
    mongolia.run_model(run_scenarios=not no_scenarios)
