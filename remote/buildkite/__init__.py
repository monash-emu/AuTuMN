import os
import sys
import logging
import subprocess as sp

import click

from autumn.constants import Region
from remote import aws

from .buildkite import trigger_pipeline
from .pipelines import (
    calibrate as calibrate_pipeline,
    full as full_pipeline,
    powerbi as powerbi_pipeline,
    trigger_philippines as trigger_philippines_pipeline,
    trigger_victoria as trigger_victoria_pipeline,
)

logger = logging.getLogger(__name__)


@click.group()
def buildkite():
    """
    CLI tool for running Buildkite jobs
    """


@buildkite.command()
def update():
    """Update Builkite pipelines to use all registered COVID models"""
    pipelines = [
        calibrate_pipeline.pipeline,
        full_pipeline.pipeline,
        powerbi_pipeline.pipeline,
        trigger_philippines_pipeline.pipeline,
        trigger_victoria_pipeline.pipeline,
    ]
    for pipeline in pipelines:
        pipeline.save()


@buildkite.command()
def calibrate():
    """Run a calibration job in Buildkite"""
    logger.info("Gathering data for calibration.")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    model_name = calibrate_pipeline.model_field.get_value()
    chains = calibrate_pipeline.chains_field.get_value()
    runtime = calibrate_pipeline.runtime_field.get_value()
    branch = calibrate_pipeline.branch_field.get_value()
    trigger_downstream = calibrate_pipeline.trigger_field.get_value()
    params_str = pprint.pformat({f.key: f.get_value() for f in calibrate_pipeline.fields}, indent=2)
    job_name = f"{model_name}-{build_number}"

    logger.info("Running calbration job %s with params:\n%s\n", job_name, params_str)
    run_id = aws.run_calibrate(
        job=job_name,
        calibration=model_name,
        chains=chains,
        runtime=runtime,
        branch=branch,
        dry=False,
    )
    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    if not trigger_downstream:
        logger.info("Not triggering full model run.")
    else:
        logger.info("Triggering full model run.")
        fp = full_pipeline
        trigger_pipeline(
            label="Trigger full model run",
            target="full-model-run",
            msg=f"Triggered by calibration {model_name} (build {build_number})",
            env={"SKIP_INPUT": "true"},
            meta={
                fp.run_id_field.key: run_id,
                fp.burn_in_field.key: fp.burn_in_field.default,
                fp.use_latest_code_field.key: fp.use_latest_code_field.default,
                fp.trigger_field.key: fp.trigger_field.default,
            },
        )

    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    logger.info("Results available at %s", get_run_url(run_id))


@buildkite.command()
def full():
    """Run a full model run job in Buildkite"""
    logger.info("Gathering data for a full model run.")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    run_id = full_pipeline.run_id_field.get_value()
    burn_in = full_pipeline.burn_in_field.get_value()
    use_latest_code = full_pipeline.use_latest_code_field.get_value()
    trigger_downstream = full_pipeline.trigger_field.get_value()
    params_str = pprint.pformat({f.key: f.get_value() for f in full_pipeline.fields}, indent=2)
    model_name, _, _ = read_run_id(run_id)
    job_name = f"{model_name}-{build_number}"

    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    logger.info("Running full model run job %s with params:\n%s\n", job_name, params_str)
    aws.run_full_model(
        job=job_name, run=run_id, burn_in=burn_in, latest_code=use_latest_code, branch="master"
    )
    if not trigger_downstream:
        logger.info("Not triggering PowerBI processing.")
    else:
        logger.info("Triggering PowerBI processing.")
        pp = powerbi_pipeline
        trigger_pipeline(
            label="Trigger PowerBI processing",
            target="powerbi-processing",
            msg=f"Triggered by full model run {model_name} (build {build_number})",
            env={"SKIP_INPUT": "true"},
            meta={pp.run_id_field.key: run_id},
        )
    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    logger.info("Results available at %s", get_run_url(run_id))


@buildkite.command()
def powerbi():
    """Run a PowerBI job in Buildkite"""
    logger.info("Gathering data for PowerBI post processing.")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    run_id = powerbi_pipeline.run_id_field.get_value()
    params_str = pprint.pformat({f.key: f.get_value() for f in powerbi_pipeline.fields}, indent=2)
    model_name, _, _ = read_run_id(run_id)
    job_name = f"{model_name}-{build_number}"

    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    logger.info("Running PowerBI post processing job %s with params:\n%s\n", job_name, params_str)
    aws.run_powerbi(job=job_name, run=run_id, branch="master")
    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
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
    _trigger_models(model_names, trigger_victoria_pipeline)


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
    _trigger_models(model_names, trigger_philippines_pipeline)


def _trigger_models(models, p):

    logger.info("Gathering data for calibration trigger.")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    chains = p.chains_field.get_value()
    runtime = p.runtime_field.get_value()
    branch = p.branch_field.get_value()
    trigger_downstream = p.trigger_field.get_value()
    params_str = pprint.pformat({f.key: f.get_value() for f in p.fields}, indent=2,)
    cp = calibrate_pipeline
    for model in models:
        logger.info("Triggering model calibration %s with params:\n%s\n", model, params_str)
        trigger_pipeline(
            label=f"Trigger calibration for {model}",
            target="calibration",
            msg=f"{model} calibration triggered by bulk run (build {build_number})",
            env={"SKIP_INPUT": "true"},
            meta={
                cp.model_field.key: model,
                cp.chains_field.key: chains,
                cp.branch_field.key: branch,
                cp.runtime_field.key: runtime,
                cp.trigger_field.key: trigger_downstream,
            },
        )


buildkite.add_command(trigger)
