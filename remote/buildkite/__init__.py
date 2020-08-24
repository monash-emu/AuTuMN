import os
import sys
import logging
import subprocess as sp

import click

from . import buildkite
from .update import update_pipelines
from remote import aws

logger = logging.getLogger(__name__)

BURN_IN_DEFAULT = 50  # Iterations


@click.group()
def buildkite():
    """
    CLI tool for running Buildkite jobs
    """


@buildkite.command()
def update():
    """Update Builkite pipelines to use all registered COVID models"""
    update_pipelines()


@buildkite.command()
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
    mcmc_branch = buildkite.get_metadata("mcmc-branch")

    # Run the calibration
    run_time_seconds = int(float(run_time_hours) * 3600)
    job_name = f"{model_name}-{build_number}"
    msg = "Running calbration job %s for %s model with %s chains for %s hours (%s seconds)"
    logger.info(msg, job_name, model_name, num_chains, run_time_hours, run_time_seconds)
    aws.run_calibrate(
        job=job_name,
        calibration=model_name,
        chains=num_chains,
        runtime=run_time_seconds,
        branch=mcmc_branch,
        dry=False,
    )
    if trigger_downstream != "yes":
        logger.info("Not triggering full model run.")
    else:
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


@buildkite.command()
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
        use_latest_code = buildkite.get_metadata("use-latest-code")
        burn_in = burn_in_option or BURN_IN_DEFAULT
        if not run_id:
            logger.error("No user-supplied `run_id` found.")
            sys.exit(1)
    else:
        # This is a job triggered by an upstream job
        logger.info("Found run id from envar: %s", run_id)
        trigger_downstream = "yes"
        use_latest_code = "no"
        burn_in = BURN_IN_DEFAULT

    # Run the full models
    model_name, _, _ = read_run_id(run_id)
    job_name = f"{model_name}-{build_number}"
    msg = "Running full model for %s with burn in %s"
    logger.info(msg, model_name, burn_in)
    aws.run_full_model(
        job=job_name,
        run=run_id,
        burn_in=burn_in,
        latest_code=use_latest_code == "yes",
        branch="master",
    )
    if trigger_downstream != "yes":
        logger.info("Not triggering PowerBI processing.")
    else:
        logger.info("Triggering PowerBI processing.")
        pipeline_data = {
            "steps": [
                {
                    "label": "Trigger PowerBI processing",
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


@buildkite.command()
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

    # Run the powerbi processing
    model_name, _, _ = read_run_id(run_id)
    job_name = f"{model_name}-{build_number}"
    logger.info("Running PowerBI post processing for model %s", model_name)
    aws.run_powerbi(job=job_name, run=run_id, branch="master")
    logger.info("PowerBI post processing for model %s suceeded", model_name)
    logger.info("Results available at %s", get_run_url(run_id))


def get_run_url(run_id: str):
    model_name, _, _ = read_run_id(run_id)
    return f"http://www.autumn-data.com/model/{run_id}/run/{run_id}.html"


def read_run_id(run_id: str):
    """Read data from run id"""
    parts = run_id.split("-")
    git_commit = parts[-1]
    timestamp = parts[-2]
    model_name = "-".join(parts[:-2])
    return model_name, timestamp, git_commit
