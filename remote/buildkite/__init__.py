import os
import sys
import logging
import subprocess as sp

import click

from autumn.constants import Region
from remote import aws

from . import buildkite
from .update import update_pipelines
from .params import CalibrateParams, FullModelRunParams, PowerBIParams

logger = logging.getLogger(__name__)


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
    params = CalibrateParams()
    job_name = f"{model_name}-{build_number}"
    msg = "Running calbration job %s for %s model with %s chains for %s seconds) on branch %s"
    logger.info(msg, job_name, params.model_name, params.chains, params.runtime, params.branch)
    run_id = aws.run_calibrate(
        job=job_name,
        calibration=params.model_name,
        chains=params.chains,
        runtime=params.runtime,
        branch=params.branch,
        dry=False,
    )
    if not params.trigger_downstream:
        logger.info("Not triggering full model run.")
    else:
        logger.info("Triggering full model run.")
        buildkite.trigger_pipeline(
            label="Trigger full model run",
            target="full-model-run",
            msg=f"Triggered by calibration {params.model_name} (build {params.buildkite.build_number})",
            env={"RUN_ID": run_id},
        )

    logger.info("Results available at %s", get_run_url(run_id))


@buildkite.command()
def full():
    """Run a full model run job in Buildkite"""
    logger.info("Starting a full model run.")
    params = FullModelRunParams()
    run_id = params.run_id
    model_name, _, _ = read_run_id(run_id)
    job_name = f"{model_name}-{params.buildkite.build_number}"
    msg = "Running full model for %s with burn in %s"
    logger.info(msg, params.model_name, params.burn_in)
    aws.run_full_model(
        job=job_name,
        run=params.run_id,
        burn_in=params.burn_in,
        latest_code=params.use_latest_code,
        branch="master",
    )
    if not params.trigger_downstream:
        logger.info("Not triggering PowerBI processing.")
    else:
        logger.info("Triggering PowerBI processing.")
        buildkite.trigger_pipeline(
            label="Trigger PowerBI processing",
            target="powerbi-processing",
            msg=f"Triggered by full model run {params.model_name} (build {params.buildkite.build_number})",
            env={"RUN_ID": run_id},
        )

    logger.info("Results available at %s", get_run_url(run_id))


@buildkite.command()
def powerbi():
    """Run a PowerBI job in Buildkite"""
    logger.info("Starting PowerBI post processing.")
    params = PowerBIParams()
    run_id = params.run_id
    model_name, _, _ = read_run_id(run_id)
    job_name = f"{model_name}-{params.buildkite.build_number}"
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


@click.group()
def trigger():
    """
    Trigger the run of run a bunch of jobs
    """


@trigger.command("victoria")
def trigger_victoria():
    """
    Trigger all Victorian models
    """
    logger.info("Triggering all Victorian regional calibrations.")
    model_names = [
        Region.NORTH_MELBOURNE,
        Region.NORTH_METRO,
        Region.SOUTH_EAST_METRO,
        Region.SOUTH_METRO,
        Region.WEST_METRO,
        Region.BARWON_SOUTH_WEST,
        Region.GIPPSLAND,
        Region.HUME,
        Region.LODDON_MALLEE,
        Region.GRAMPIANS,
    ]
    _trigger_models(model_names)


@trigger.command("philippines")
def trigger_philippines():
    """
    Trigger all Philippines models
    """
    logger.info("Triggering all Philippines regional calibrations.")
    model_names = [
        Region.PHILIPPINES,
        Region.MANILA,
        Region.CALABARZON,
        Region.CENTRAL_VISAYAS,
    ]
    _trigger_models(model_names)


def _trigger_models(models):
    params = CalibrateParams()
    for model in models:
        buildkite.trigger_pipeline(
            label=f"Trigger calibration for {model}",
            target="calibration",
            msg=f"{model} calibration triggered by bulk run (build {params.buildkite.build_number})",
            meta={
                "model-name": model,
                "trigger-downstream": params.trigger_downstream,
                "num-chains": params.chains,
                "mcmc-runtime": params.runtime,
                "mcmc-branch": params.branch,
            },
        )


buildkite.add_command(trigger)

