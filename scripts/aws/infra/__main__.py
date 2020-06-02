import time

import click

from . import aws
from . import remote


@click.group()
def cli():
    """
    Simple AWS EC2 instance manager
    """


@click.command()
def status():
    """Print EC2 instance status"""
    instances = aws.describe_instances()
    aws.print_status(instances)


@click.command()
@click.argument("job_id")
def start(job_id):
    """
    Start a job but don't stop it
    """
    aws.run_job(job_id)


@click.command()
@click.argument("job_id")
def stop(job_id):
    """
    Stop a job
    """
    aws.stop_job(job_id)


@click.command()
def cleanup():
    """
    Cleanup dangling AWS bits.
    """
    aws.cleanup_volumes()


@click.command()
@click.argument("name")
def ssh(name):
    """SSH into an EC2 instance"""
    instance = aws.find_instance(name)
    if instance and aws.is_running(instance):
        remote.ssh_interactive(instance)
    elif instance:
        click.echo(f"Instance {name} not running")
    else:
        click.echo(f"Instance {name} not found")


@click.group()
def run():
    """
    Run a job
    """


@run.command("calibrate")
@click.argument("job_name", type=str)
@click.argument("calibration_name", type=str)
@click.argument("num_chains", type=int)
@click.argument("run_time", type=int)
def run_calibrate(job_name, calibration_name, num_chains, run_time):
    """
    Run a MCMC calibration
    """
    job_id = f"calibrate-{job_name}"
    script_args = [calibration_name, num_chains, run_time]
    _run_job(job_id, "run_calibrate.sh", script_args)


@run.command("full")
@click.argument("job_name", type=str)
@click.argument("run_name", type=str)
@click.argument("burn_in", type=int)
def run_full_model(job_name, run_name, burn_in):
    """
    Run the full models based off an MCMC calibration
    """
    job_id = f"full-{job_name}"
    script_args = [run_name, burn_in]
    _run_job(job_id, "run_full_model.sh", script_args)


@run.command("powerbi")
@click.argument("job_name", type=str)
@click.argument("run_name", type=str)
def run_powerbi(job_name, run_name):
    """
    Run the collate a PowerBI database from the full model run outputs.
    """
    job_id = f"powerbi-{job_name}"
    script_args = [run_name]
    _run_job(job_id, "run_powerbi.sh", script_args)


def _run_job(job_id, script_name, script_args):
    """
    Run a job on a remote server
    """
    try:
        aws.run_job(job_id)
    except aws.NoInstanceAvailable:
        click.echo("Could not run job - no instances available")
        return

    print("Waiting 60s for server to boot... ", end="", flush=True)
    time.sleep(60)
    print("done.")
    instance = aws.find_instance(job_id)
    start_time = time.time()
    while time.time() - start_time < 20:
        print("Attempting to run job")
        remote.ssh_run_job(instance, script_name, script_args)
        time.sleep(3)

    aws.stop_job(job_id)


cli.add_command(cleanup)
cli.add_command(status)
cli.add_command(ssh)
cli.add_command(start)
cli.add_command(stop)
cli.add_command(run)
cli()
