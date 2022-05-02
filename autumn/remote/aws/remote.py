import logging
import os
import subprocess

from fabric import Connection

from autumn.tools.utils.runs import build_run_id, read_run_id

from .runner import SSHRunner, get_connection

logger = logging.getLogger(__name__)


def cleanup_builds(instance):
    with get_connection(instance) as conn:
        conn.run("sudo rm -rf /var/lib/buildkite-agent/builds/", echo=True)


def run_powerbi(runner: SSHRunner, run_id: str, urunid: str, commit: str):
    """Run PowerBI processing on the remote server"""
    run_id = run_id.lower()
    msg = "Running PowerBI processing for run %s on AWS instance %s"
    logger.info(msg, run_id, runner.instance["InstanceId"])

    runner.print_hostname()
    if commit != "use_original_commit":
        runner.set_repo_to_commit(commit=commit)
    else:
        runner.set_run_id(run_id)

    runner.install_requirements()
    runner.read_secrets()
    pipeline_name = "powerbi"
    pipeline_args = {"run": run_id, "urunid": urunid}
    runner.run_task_pipeline(pipeline_name, pipeline_args)
    logger.info("PowerBI processing completed for %s", run_id)


def run_full_model(
    runner: SSHRunner, run_id: str, burn_in: int, sample: int, commit: str
):
    """Run full model job on the remote server"""
    run_id = run_id.lower()
    msg = "Running full models for run %s burn-in %s on AWS instance %s"
    logger.info(msg, run_id, burn_in, runner.instance["InstanceId"])

    runner.print_hostname()
    if commit != "use_original_commit":
        runner.set_repo_to_commit(commit=commit)
    else:
        runner.set_run_id(run_id)

    runner.install_requirements()
    runner.read_secrets()
    pipeline_name = "full"
    pipeline_args = {
        "run": run_id,
        "burn": burn_in,
        "sample": sample,
    }
    runner.run_task_pipeline(pipeline_name, pipeline_args)
    logger.info("Full model runs completed for %s", run_id)


def resume_calibration(
    runner: SSHRunner,
    baserun: str,
    num_chains: int,
    runtime: int,
):
    """Resume calibration job on the remote server"""
    msg = "Resuming calibration with %s chains for %s seconds on AWS instance %s."
    logger.info(msg, num_chains, runtime, runner.instance["InstanceId"])
    run_id = None

    app_name, region_name, _, orig_commit = baserun.split('/')

    runner.print_hostname()
    runner.set_run_id( baserun)
    runner.install_requirements()
    runner.read_secrets()
    run_id = runner.get_run_id(app_name, region_name)
    pipeline_name = "resume_calibration"
    pipeline_args = {
        "run": run_id,
        "baserun": baserun,
        "chains": num_chains,
        "runtime": runtime,
    }
    runner.run_task_pipeline(pipeline_name, pipeline_args)
    logger.info("Calibration resume completed for %s", run_id)

    return run_id

def run_calibration(
    runner: SSHRunner,
    app_name: str,
    region_name: str,
    num_chains: int,
    runtime: int,
    commit: str,
):
    """Run calibration job on the remote server"""
    msg = "Running calibration %s %s with %s chains for %s seconds on AWS instance %s."
    logger.info(msg, app_name, region_name, num_chains, runtime, runner.instance["InstanceId"])
    run_id = None
    
    runner.print_hostname()
    runner.set_repo_to_commit(commit=commit)
    runner.install_requirements()
    runner.read_secrets()
    run_id = runner.get_run_id(app_name, region_name)
    pipeline_name = "calibrate"
    pipeline_args = {
        "run": run_id,
        "chains": num_chains,
        "runtime": runtime,
    }
    runner.run_task_pipeline(pipeline_name, pipeline_args)
    logger.info("Calibration completed for %s", run_id)

    return run_id



LOGS_URL = "https://monashemu.grafana.net/explore?orgId=1&left=%5B%22now-1h%22,%22now%22,%22grafanacloud-monashemu-logs%22,%7B%22expr%22:%22%7Bjob%3D%5C%22app%5C%22%7D%20%7C%3D%20%5C%22${HOSTNAME}%5C%22%22%7D%5D"
METRICS_URL = "https://monashemu.grafana.net/d/SkUUUHyMk/nodes?orgId=1&refresh=30s&var-datasource=grafanacloud-monashemu-prom&var-instance=${HOSTNAME}:12345"


