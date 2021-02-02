import time
import os
import sys
import logging
import functools
from typing import List

import click
from botocore.exceptions import ClientError
from invoke.exceptions import UnexpectedExit

from . import aws
from . import remote
from settings import EC2InstanceType, EC2_INSTANCE_SPECS

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


PROTECTED_INSTANCES = [
    "buildkite",
    "buildkite-1",
    "buildkite-2",
    "buildkite-3",
]


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
    aws.cleanup_instances()
    aws.cleanup_volumes()


@aws_cli.command()
def cleanup_builds():
    """
    Cleanup Buildkite builds.
    """
    for name in ["buildkite-1", "buildkite-1", "buildkite-3"]:
        logging.info("Cleaning up builds for %s", name)
        instance = aws.find_instance(name)
        remote.cleanup_builds(instance)


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
@click.option("--app", type=str, required=True)
@click.option("--region", type=str, required=True)
@click.option("--chains", type=int, required=True)
@click.option("--runtime", type=int, required=True)
@click.option("--branch", type=str, default="master")
@click.option("--spot", is_flag=True)
@click.option("--dry", is_flag=True)
def run_calibrate_cli(job, app, region, chains, runtime, branch, spot, dry):
    run_calibrate(job, app, region, chains, runtime, branch, spot, dry)


def run_calibrate(job, app, region, chains, runtime, branch, is_spot, dry):
    """
    Run a MCMC calibration on an AWS server.
    """
    job_id = f"calibrate-{job}"
    if region == "victoria" and app == "covid_19":
        instance_type = aws.get_instance_type(2 * chains, 8)
    else:
        instance_type = aws.get_instance_type(chains, 8)

    if dry:
        logger.info("Dry run, would have used instance type: %s", instance_type)
    else:
        kwargs = {
            "num_chains": chains,
            "app_name": app,
            "region_name": region,
            "runtime": runtime,
            "branch": branch,
        }
        job_func = functools.partial(remote.run_calibration, **kwargs)
        return _run_job(job_id, [instance_type], is_spot, job_func)


@run.command("full")
@click.option("--job", type=str, required=True)
@click.option("--run", type=str, required=True)
@click.option("--burn-in", type=int, required=True)
@click.option("--latest-code", is_flag=True)
@click.option("--branch", type=str, default="master")
@click.option("--spot", is_flag=True)
def run_full_model_cli(job, run, burn_in, latest_code, branch, spot):
    run_full_model(job, run, burn_in, latest_code, branch, spot)


def run_full_model(job, run, burn_in, latest_code, branch, is_spot):
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
    _run_job(job_id, [instance_type], is_spot, job_func)


@run.command("powerbi")
@click.option("--job", type=str, required=True)
@click.option("--run", type=str, required=True)
@click.option("--branch", type=str, default="master")
@click.option("--spot", is_flag=True)
def run_powerbi_cli(job, run, branch, spot):
    run_powerbi(job, run, branch, spot)


def run_powerbi(job, run, branch, is_spot):
    """
    Run the collate a PowerBI database from the full model run outputs.
    """
    job_id = f"powerbi-{job}"
    instance_types = [
        EC2InstanceType.r5_8xlarge,
        EC2InstanceType.r5d_8xlarge,
        EC2InstanceType.r5a_8xlarge,
        EC2InstanceType.r5a_16xlarge,
    ]
    kwargs = {"run_id": run, "branch": branch}
    job_func = functools.partial(remote.run_powerbi, **kwargs)
    _run_job(job_id, instance_types, is_spot, job_func)


@run.command("dhhs")
@click.option("--job", type=str, required=True)
@click.option("--commit", type=str, required=True)
@click.option("--branch", type=str, default="master")
@click.option("--spot", is_flag=True)
def run_dhhs_cli(job, commit, branch, spot):
    run_dhhs(job, commit, branch, spot)


def run_dhhs(job, commit, branch, is_spot):
    """
    Perform DHHS post processing.
    """
    job_id = f"dhhs-{job}"
    instance_type = EC2InstanceType.m5_16xlarge
    kwargs = {"commit": commit, "branch": branch}
    job_func = functools.partial(remote.run_dhhs, **kwargs)
    _run_job(job_id, [instance_type], is_spot, job_func)


def _run_job(job_id: str, instance_types: List[str], is_spot: bool, job_func):
    """
    Run a job on a remote server
    """
    aws_client_exc = None
    for instance_type in instance_types:
        aws_client_exc = None
        try:
            aws.run_job(job_id, instance_type, is_spot)
            logger.info("Waiting 60s for %s server to boot... ", instance_type)
            time.sleep(60)
            logger.info("Server is hopefully ready.")
            break
        except ClientError as e:
            # Issue starting the server, eg. insufficient spot capacity.
            aws_client_exc = e
            logger.error(
                "Failed to start AWS %s job for instance type %s. %s: %s",
                "spot" if is_spot else "non-spot",
                instance_type,
                e.__class__.__name__,
                e,
            )

    if aws_client_exc:
        raise aws_client_exc

    instance = aws.find_instance(job_id)
    return_value = None
    try:
        logger.info("Attempting to run job %s on instance %s", job_id, instance["InstanceId"])
        return_value = job_func(instance=instance)
        logging.info("Job %s succeeded.", job_id)
    except UnexpectedExit as e:
        # Invoke error - happened when commands running on remote machine.
        # We will see a better stack trace from the SSH output, no need to print this.
        logger.error(
            "Job %s failed when running a command on the remote machine. %s: %s",
            job_id,
            e.__class__.__name__,
            e,
        )
        sys.exit(-1)
    except Exception as e:
        # Unknown error.
        logger.exception(f"Job {job_id} failed.")
        raise e
    finally:
        # Always stop the job to prevent dangling jobs.
        aws.stop_job(job_id)

    return return_value


aws_cli.add_command(run)
