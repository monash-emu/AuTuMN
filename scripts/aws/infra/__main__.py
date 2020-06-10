import time
import os

import click

from . import aws
from . import remote
from .website import update_website
from .settings import EC2InstanceType


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
@click.argument("run_name")
def logs(run_name):
    """Get all logs for a given run"""
    s3_key = f"s3://autumn-calibrations/{run_name}/logs"
    dest = f"logs/{run_name}"
    os.makedirs(dest, exist_ok=True)
    aws.download_s3(s3_key, dest)


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
@click.option("--dry", is_flag=True)
def run_calibrate(job_name, calibration_name, num_chains, run_time, dry):
    """
    Run a MCMC calibration
    """
    job_id = f"calibrate-{job_name}"
    script_args = [calibration_name, num_chains, run_time]
    instance_type = aws.get_instance_type(num_chains, 8)
    if dry:
        print("Dry run:", instance_type)
    else:
        _run_job(job_id, instance_type, "run_calibrate.sh", script_args)


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
    instance_type = EC2InstanceType.m5_8xlarge
    instance_type = aws.get_instance_type(30, 8)
    _run_job(job_id, instance_type, "run_full_model.sh", script_args)


@run.command("powerbi")
@click.argument("job_name", type=str)
@click.argument("run_name", type=str)
def run_powerbi(job_name, run_name):
    """
    Run the collate a PowerBI database from the full model run outputs.
    """
    job_id = f"powerbi-{job_name}"
    script_args = [run_name]
    instance_type = aws.get_instance_type(30, 32)
    _run_job(job_id, instance_type, "run_powerbi.sh", script_args)


def _run_job(job_id, instance_type, script_name, script_args):
    """
    Run a job on a remote server
    """
    try:
        aws.run_job(job_id, instance_type)
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


@click.command()
def website():
    """
    Update the calibrations website.
    """
    update_website()


cli.add_command(website)
cli.add_command(logs)
cli.add_command(cleanup)
cli.add_command(status)
cli.add_command(ssh)
cli.add_command(start)
cli.add_command(stop)
cli.add_command(run)
cli()
