import os
import sys
import logging
import subprocess as sp

import click

from . import buildkite

logger = logging.getLogger(__name__)

BURN_IN_DEFAULT = 1000 # Iterations

@click.group()
def cli():
    """
    CLI tool for running Buildkite jobs
    """


@click.command()
def calibrate():
    """Run a calibration job in Buildkite"""
    logger.info("Starting calibration.")
    # Pull in envars
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    commit = os.environ["BUILDKITE_COMMIT"]
    branch = os.environ["BUILDKITE_BRANCH"]
    # Pull in metadata
    trigger_downstream = buildkite.get_metadata("trigger-downstream")
    model_name = buildkite.get_metadata("model-name")
    num_chains = buildkite.get_metadata("mcmc-num-chains")
    run_time_hours = buildkite.get_metadata("mcmc-runtime")
    # Run the calibration
    run_time_seconds = int(run_time_hours * 3600)
    job_name = f"{model_name}-{build_number}"
    msg = "Running calbration job %s for %s model with %s chains for %s hours (%s seconds)"
    logger.info(msg, job_name, model_name, num_chains, run_time_hours, run_time_seconds)
    try:
        cmd_str = (
            "scripts/aws/run.sh run calibrate"
            f" --job {job_name}"
            f" --calibration {model_name}"
            f" --chains {num_chains}"
            f" --runtime {run_time_seconds}"
        )
        proc = sp.run(cmd_str, shell=True, check=True, stdout=sp.PIPE, encoding="utf-8")
        # Get `run_id` from string with format "Calibration completed for $RUN_ID"
        run_id = None
        lines = (l.strip() for l in proc.stdout.split("\n"))
        for l in lines:
            if l.startswith("Calibration completed for "):
                run_id = l.split(" ")[-1]

        if not run_id:
            raise ValueError("Could not find `run_id` in stdout")

    except Exception:
        logger.exception("Calibration for job %s failed", job_name)
        sys.exit(1)

    logging.info("Calibration for job %s succeeded", job_name)
    if trigger_downstream != "yes":
        logger.info("Not triggering full model run.")
        return

    logger.info("Triggering full model run.")
    pipeline_data = {
        "steps": [
            {
                "label": "Trigger full model run",
                "trigger": "full-model-run",
                "async": True,
                "build": {
                    "message": f"Triggered by calibration {model_name} (build {build_number})",
                    "commit": commit,
                    "branch": branch,
                    "env": {"RUN_ID": run_id},
                },
            }
        ]
    }
    buildkite.trigger_pipeline(pipeline_data)
    logger.info("Results available at %s", get_run_url(run_id))


@click.command()
def full():
    """Run a full model run job in Buildkite"""
    logger.info("Starting a full model run.")
    # Pull in envars
    run_id = os.environ.get("RUN_ID")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    commit = os.environ["BUILDKITE_COMMIT"]
    branch = os.environ["BUILDKITE_BRANCH"]
    if not run_id:
        # Pull in user-supplied metadata
        logger.info("Using user-supplied run name.")
        burn_in_option = buildkite.get_metadata("full-burn-in")
        run_id = buildkite.get_metadata("run-id")
        trigger_downstream = buildkite.get_metadata("trigger-downstream")
        burn_in = burn_in_option or  BURN_IN_DEFAULT
        if not run_id:
            logger.error("No user-supplied `run_id` found.")
            sys.exit(1)
    else:
        # This is a job triggered by an upstream job
        logger.info("Found run id from envar: %s", run_id)
        trigger_downstream = "yes"
        burn_in = BURN_IN_DEFAULT

    # Run the full models
    model_name, _, _ = read_run_id(run_id)
    job_name= f"{model_name}-{build_number}""
    msg = "Running full model for %s with burn in %s"
    logger.info(msg, model_name, burn_in)
    try:
        cmd_str = (
            "scripts/aws/run.sh run full"
            f" --job {job_name}"
            f" --run {run_id}"
            f" --burn-in {burn_in}"
        )
        proc = sp.run(cmd_str, shell=True, check=True, stdout=sp.PIPE, encoding="utf-8")
    except Exception:
        logger.exception("Full model run for job %s, run id %s failed", job_name, run_id)
        sys.exit(1)

    logging.info("Calibration for job %s succeeded", job_name)
    if trigger_downstream != "yes":
        logger.info("Not triggering PowerBI processing.")
        return

    logger.info("Triggering PowerBI processing.")
    pipeline_data = {
        "steps": [
            {
                "label": "Trigger full model run",
                "trigger": "powerbi-processing",
                "async": True,
                "build": {
                    "message": f"Triggered by full model run {model_name} (build {build_number})",
                    "commit": commit,
                    "branch": branch,
                    "env": {"RUN_ID": run_id},
                },
            }
        ]
    }
    buildkite.trigger_pipeline(pipeline_data)
    logger.info("Results available at %s", get_run_url(run_id))



@click.command()
def powerbi():
    """Run a PowerBI job in Buildkite"""
    logger.info("Starting PowerBI post processing.")
    # Pull in envars
    run_id = os.environ.get("RUN_ID")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    if not run_id:
        # Pull in user-supplied metadata
        logger.info("Using user-supplied run name.")
        run_id = buildkite.get_metadata("run-id")
        if not run_id:
            logger.error("No user-supplied `run_id` found.")
            sys.exit(1)
    else:
        # This is a job triggered by an upstream job
        logger.info("Found run id from envar: %s", run_id)

    # Run the full models
    model_name, _, _ = read_run_id(run_id)
    job_name= f"{model_name}-{build_number}""
    logger.info("Running PowerBI post processing for model %s", model_name)
    try:
        cmd_str = (
            "scripts/aws/run.sh run powerbi"
            f" --job {job_name}"
            f" --run {run_id}"
        )
        proc = sp.run(cmd_str, shell=True, check=True, stdout=sp.PIPE, encoding="utf-8")
    except Exception:
        logger.info("Failed to run PowerBI post processing for run: %s", run_id)
        sys.exit(1)

    logger.info("PowerBI post processing for model %s suceeded", model_name)
    logger.info("Results available at %s", get_run_url(run_id))
    

def get_run_url(run_id: str):
    return f"http://autumn-data.s3-website-ap-southeast-2.amazonaws.com/{run_id}"


def read_run_id(run_id: str):
    """Read data from run id"""
    parts = run_id.split("-")
    git_commit = parts[-1]
    timestamp = parts[-2]
    model_name = "-".join(parts[:-2])
    return model_name, timestamp, git_commit

cli.add_command(calibrate)
cli.add_command(full)
cli.add_command(powerbi)
cli()
