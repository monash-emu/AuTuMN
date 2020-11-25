"""
Runs AuTuMN apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from apps import covid_19, sir_example, tuberculosis, tuberculosis_strains


@click.group()
def run():
    """Run a model"""


@run.command("tbs")
@click.argument("region", type=click.Choice(tuberculosis_strains.app.region_names))
@click.option("--no-scenarios", is_flag=True)
def run_covid(region, no_scenarios):
    """Run the tuberculosis_strains model for some region"""
    app_region = tuberculosis_strains.app.get_region(region)
    app_region.run_model(run_scenarios=not no_scenarios)


@run.command("tb")
@click.argument("region", type=click.Choice(tuberculosis.app.region_names))
@click.option("--no-scenarios", is_flag=True)
def run_covid(region, no_scenarios):
    """Run the tuberculosis model for some region"""
    app_region = tuberculosis.app.get_region(region)
    app_region.run_model(run_scenarios=not no_scenarios)


@run.command("covid")
@click.argument("region", type=click.Choice(covid_19.app.region_names))
@click.option("--no-scenarios", is_flag=True)
def run_covid(region, no_scenarios):
    """Run the COVID model for some region"""
    app_region = covid_19.app.get_region(region)
    app_region.run_model(run_scenarios=not no_scenarios)


@run.command("example")
@click.argument("region", type=click.Choice(sir_example.app.region_names))
@click.option("--no-scenarios", is_flag=True)
def run_sir_example(region, no_scenarios):
    """Run the SIR example model for some region"""
    app_region = sir_example.app.get_region(region)
    app_region.run_model(run_scenarios=not no_scenarios)
