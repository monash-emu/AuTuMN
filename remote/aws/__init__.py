import time
import os
import sys
import logging
import click
import functools

from . import aws
from . import remote
from .settings import EC2InstanceType, EC2_INSTANCE_SPECS

logger = logging.getLogger(__name__)


@click.group()
def aws_cli():
    """
    CLI tool to run jobs on AWS EC2 instances
    """


@aws_cli.command()
def status():
    """Print EC2 instance status"""
    instances = aws.describe_instances()
    aws.print_status(instances)


@aws_cli.command()
@click.argument("job_id")
@click.argument("instance_type", type=click.Choice(EC2_INSTANCE_SPECS.keys()))
def start(job_id, instance_type):
    """
    Start a job but don't stop it
    """
    aws.run_job(job_id, instance_type)


PROTECTED_INSTANCES = ["buildkite"]


@aws_cli.command()
@click.argument("job_id")
def stop(job_id):
    """
    Stop a job, stops all jobs if "all"
    """
    if job_id == "all":
        for i in aws.describe_instances():
            if i["name"] not in PROTECTED_INSTANCES:
                job_id = i["name"]
                aws.stop_job(job_id)

    else:
        aws.stop_job(job_id)


@aws_cli.command()
def cleanup():
    """
    Cleanup dangling AWS bits.
    """
    aws.cleanup_volumes()
    aws.cleanup_instances()


@aws_cli.command()
@click.argument("run_name")
def logs(run_name):
    """Get all logs for a given run"""
    s3_key = f"s3://autumn-data/{run_name}/logs"
    dest = f"logs/{run_name}"
    os.makedirs(dest, exist_ok=True)
    aws.download_s3(s3_key, dest)


@aws_cli.command()
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
@click.option("--job", type=str, required=True)
@click.option("--calibration", type=str, required=True)
@click.option("--chains", type=int, required=True)
@click.option("--runtime", type=int, required=True)
@click.option("--branch", type=str, default="master")
@click.option("--dry", is_flag=True)
def run_calibrate_cli(job, calibration, chains, runtime, branch, dry):
    run_calibrate(job, calibration, chains, runtime, branch, dry)


def run_calibrate(job, calibration, chains, runtime, branch, dry):
    """
    Run a MCMC calibration on an AWS server.
    """
    job_id = f"calibrate-{job}"
    instance_type = aws.get_instance_type(chains, 8)
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
        return _run_job(job_id, instance_type, job_func)


@run.command("full")
@click.option("--job", type=str, required=True)
@click.option("--run", type=str, required=True)
@click.option("--burn-in", type=int, required=True)
@click.option("--latest-code", is_flag=True)
@click.option("--branch", type=str, default="master")
def run_full_model_cli(job, run, burn_in, latest_code, branch):
    run_full_model(job, run, burn_in, latest_code, branch)


def run_full_model(job, run, burn_in, latest_code, branch):
    """
    Run the full models based off an MCMC calibration on an AWS server.
    """
    job_id = f"full-{job}"
    instance_type = EC2InstanceType.r5_2xlarge
    kwargs = {
        "run_id": run,
        "burn_in": burn_in,
        "use_latest_code": latest_code,
        "branch": branch,
    }
    job_func = functools.partial(remote.run_full_model, **kwargs)
    _run_job(job_id, instance_type, job_func)


@run.command("powerbi")
@click.option("--job", type=str, required=True)
@click.option("--run", type=str, required=True)
@click.option("--branch", type=str, default="master")
def run_powerbi_cli(job, run, branch):
    run_powerbi(job, run, branch)


def run_powerbi(job, run, branch):
    """
    Run the collate a PowerBI database from the full model run outputs.
    """
    job_id = f"powerbi-{job}"
    instance_type = EC2InstanceType.r5_2xlarge
    kwargs = {"run_id": run, "branch": branch}
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
        sys.exit(-1)

    logger.info("Waiting 60s for %s server to boot... ", instance_type)
    time.sleep(60)
    logger.info("Server is hopefully ready.")
    instance = aws.find_instance(job_id)
    return_value = None
    try:
        logger.info("Attempting to run job %s on instance %s", job_id, instance["InstanceId"])
        return_value = job_func(instance=instance)
        logging.info("Job %s succeeded.", job_id)
    except Exception:
        logger.error(f"Running job {job_id} failed")
        raise
    finally:
        aws.stop_job(job_id)

    return return_value


aws_cli.add_command(run)
