"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from apps import covid_19, sir_example


@click.group()
def run():
    """Run a model"""


@run.command("covid")
@click.argument("region", type=click.Choice(covid_19.app.region_names))
@click.option("--no-scenarios", is_flag=True)
def run_covid(region, no_scenarios):
    """Run the COVID model for some region"""
    covid_region = covid_19.app.get_region(region)
    covid_region.run_model(run_scenarios=not no_scenarios)


@run.command("example")
@click.argument("region", type=click.Choice(sir_example.app.region_names))
@click.option("--no-scenarios", is_flag=True)
def run_sir_example(region, no_scenarios):
    """Run the SIR example model for some region"""
    sir_region = sir_example.app.get_region(region)
    sir_region.run_model(run_scenarios=not no_scenarios)
