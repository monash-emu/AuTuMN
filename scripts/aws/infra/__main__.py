import time
import os
import logging
import click
import functools

from . import aws
from . import remote
from .website import update_website
from .settings import EC2InstanceType, EC2_INSTANCE_SPECS

logger = logging.getLogger(__name__)


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
@click.argument("instance_type", type=click.Choice(EC2_INSTANCE_SPECS.keys()))
def start(job_id, instance_type):
    """
    Start a job but don't stop it
    """
    aws.run_job(job_id, instance_type)


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
    aws.cleanup_instances()


@click.command()
@click.argument("run_name")
def logs(run_name):
    """Get all logs for a given run"""
    s3_key = f"s3://autumn-data/{run_name}/logs"
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


@click.command()
def website():
    """
    Update the calibrations website.
    """
    update_website()


@click.group()
def run():
    """
    Run a job
    """


@run.command("calibrate")
@click.option("--job", type=str, required=True)
@click.option("--calibration", type=str, required=True)
@click.option("--chains", type=int, required=True)
@click.option("--runtime", type=int, required=True)
@click.option("--branch", type=str, default="master")
@click.option("--dry", is_flag=True)
def run_calibrate(job, calibration, chains, runtime, branch, dry):
    """
    Run a MCMC calibration on an AWS server.
    Example usage:

        python -m infra run calibrate \
        --job test \
        --calibration malaysia \
        --chains 6 \
        --runtime 200 \
        --branch luigi-redux

    """
    job_id = f"calibrate-{job}"
    instance_type = aws.get_instance_type(2 * chains, 8)
    if dry:
        logger.info("Dry run, would have used instance type: %s", instance_type)
    else:
        kwargs = {
            "num_chains": chains,
            "model_name": calibration,
            "runtime": runtime,
            "branch": branch,
        }
        job_func = functools.partial(remote.run_calibration, **kwargs)
        _run_job(job_id, instance_type, job_func)


@run.command("full")
@click.option("--job", type=str, required=True)
@click.option("--run", type=str, required=True)
@click.option("--burn-in", type=int, required=True)
@click.option("--latest-code", is_flag=True)
def run_full_model(job, run, burn_in, latest_code):
    """
    Run the full models based off an MCMC calibration on an AWS server.
    Example usage:

        python -m infra run full \
        --run malaysia-1594104927-master-a22f1435b6989f607afca25e4ea59273e33a962b \
        --job test \
        --burn-in 1000

    """
    job_id = f"full-{job}"
    instance_type = aws.get_instance_type(30, 8)
    kwargs = {
        "run_id": run,
        "burn_in": burn_in,
        "use_latest_code": latest_code,
    }
    job_func = functools.partial(remote.run_full_model, **kwargs)
    _run_job(job_id, instance_type, job_func)


@run.command("powerbi")
@click.option("--job", type=str, required=True)
@click.option("--run", type=str, required=True)
def run_powerbi(job, run):
    """
    Run the collate a PowerBI database from the full model run outputs.

        python -m infra run full \
        --run malaysia-1594104927-master-a22f1435b6989f607afca25e4ea59273e33a962b \
        --job test

    """
    job_id = f"powerbi-{job}"
    instance_type = aws.get_instance_type(30, 32)
    kwargs = {"run_id": run}
    job_func = functools.partial(remote.run_powerbi, **kwargs)
    _run_job(job_id, instance_type, job_func)


def _run_job(job_id, instance_type, job_func):
    """
    Run a job on a remote server
    """
    try:
        aws.run_job(job_id, instance_type)
    except aws.NoInstanceAvailable:
        click.echo("Could not run job - no instances available")
        return

    logger.info("Waiting 60s for %s server to boot... ", instance_type)
    time.sleep(60)
    logger.info("Server is hopefully ready.")
    instance = aws.find_instance(job_id)
    try:
        logger.info("Attempting to run job %s on instance %s", job_id, instance["InstanceId"])
        job_func(instance=instance)
    except Exception as e:
        logger.exception("Running job failed")
        raise
    finally:
        aws.stop_job(job_id)


cli.add_command(website)
cli.add_command(logs)
cli.add_command(cleanup)
cli.add_command(status)
cli.add_command(ssh)
cli.add_command(start)
cli.add_command(stop)
cli.add_command(run)
cli()
