"""
Runs Autumn apps

You can access this script from your CLI by running:

    python -m apps --help

"""
import click

from autumn.core.registry import get_registered_model_names, get_registered_project_names


@click.group()
def project():
    """Project commands."""


@project.command("run")
@click.argument("model", type=click.Choice(get_registered_model_names()))
@click.argument("project", type=click.Choice(get_registered_project_names()))
@click.option("--no-scenarios", is_flag=True)
def run_model(model, project, no_scenarios):
    """Run a model for some project"""
    from autumn.core.project import get_project, run_project_locally

    project = get_project(model, project)
    run_project_locally(project, run_scenarios=not no_scenarios)


@project.command("calibrate")
@click.argument("model", type=click.Choice(get_registered_model_names()))
@click.argument("project", type=click.Choice(get_registered_project_names()))
@click.argument("max_seconds", type=int)
@click.argument("run_id", type=int)
@click.option("--num-chains", type=int, default=1)
def calibrate_model(model, project, max_seconds, run_id, num_chains):
    """Calibrate a model for some project"""
    from autumn.core.project import get_project

    project = get_project(model, project)
    project.calibrate(max_seconds, run_id, num_chains)


@project.command("plotrmi")
def plotting_project():
    """Plot all model outputs for the Marshall Islands project"""
    from autumn.projects.tuberculosis.marshall_islands.outputs.main_script import make_all_rmi_plots

    make_all_rmi_plots()


@project.command("runsampleopti")
def run_sample():
    """Run sample code for optimisation"""
    from autumn.projects.covid_19.mixing_optimisation.sample_code import run_sample_code

    run_sample_code()


@project.command("runsamplevaccopti")
def run_sample():
    """Run sample code for optimisation"""
    from autumn.projects.covid_19.vaccine_optimisation.sample_code import run_sample_code

    run_sample_code()


@project.command("run_vaccopti_scenario_write")
def run_vaccopti_scenario_write():
    from autumn.projects.covid_19.vaccine_optimisation.utils import write_optimised_scenario

    write_optimised_scenario()
