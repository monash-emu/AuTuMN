import logging
import os
import pprint

import click

from autumn.settings import Region
from autumn.infrastructure.remote.aws import cli as aws
from autumn.tools.utils.runs import read_run_id

from .buildkite import trigger_pipeline
from .pipelines import calibrate as calibrate_pipeline
from .pipelines import resume as resume_pipeline
from .pipelines import full as full_pipeline
from .pipelines import powerbi as powerbi_pipeline
from .pipelines import trigger_europe as trigger_europe_pipeline
from .pipelines import trigger_philippines as trigger_philippines_pipeline
from .pipelines import trigger_victoria as trigger_victoria_pipeline
from .pipelines import trigger_malaysia as trigger_malaysia_pipeline

logger = logging.getLogger(__name__)


@click.group()
def buildkite_cli():
    """
    CLI tool for running Buildkite jobs
    """


@buildkite_cli.command()
def update():
    """Update Builkite pipelines to use all registered COVID models"""
    pipelines = [
        calibrate_pipeline.pipeline,
        full_pipeline.pipeline,
        powerbi_pipeline.pipeline,
        resume_pipeline.pipeline,
        trigger_philippines_pipeline.pipeline,
        trigger_victoria_pipeline.pipeline,
        trigger_europe_pipeline.pipeline,
        trigger_malaysia_pipeline.pipeline,
    ]
    for pipeline in pipelines:
        pipeline.save()


@buildkite_cli.command()
def calibrate():
    """Run a calibration job in Buildkite"""
    logger.info("Gathering data for calibration.")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    model_name = calibrate_pipeline.model_field.get_value()
    chains = calibrate_pipeline.chains_field.get_value()
    runtime = calibrate_pipeline.runtime_field.get_value()
    burn_in = calibrate_pipeline.burn_in_field.get_value()
    sample_size = calibrate_pipeline.sample_size_field.get_value()
    commit = calibrate_pipeline.commit_field.get_value()
    trigger_downstream = calibrate_pipeline.trigger_field.get_value()
    params_str = pprint.pformat({f.key: f.get_value() for f in calibrate_pipeline.fields}, indent=2)

    # Decode combined app + model name from user input.
    app_name, region_name = model_name.split(":")
    job_name = f"{app_name}-{region_name}-{build_number}"

    logger.info("Running calibration job %s with params:\n%s\n", job_name, params_str)
    run_id = aws.run_calibrate(
        job=job_name,
        app=app_name,
        region=region_name,
        chains=chains,
        runtime=runtime,
        commit=commit,
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
            msg=f"Triggered by calibration {app_name} {region_name} (build {build_number})",
            env={"SKIP_INPUT": "true"},
            meta={
                fp.run_id_field.key: run_id,
                fp.burn_in_field.key: burn_in,
                fp.sample_size_field.key: sample_size,
                fp.commit_field.key: fp.commit_field.default,
                fp.trigger_field.key: fp.trigger_field.get_option(trigger_downstream),
            },
        )

    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    logger.info("Results available at %s", get_run_url(run_id))

@buildkite_cli.command()
def resume():
    """Run a calibration job in Buildkite"""
    logger.info("Gathering data for calibration.")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    baserun = resume_pipeline.run_id_field.get_value()
    chains = resume_pipeline.chains_field.get_value()
    runtime = resume_pipeline.runtime_field.get_value()
    burn_in = resume_pipeline.burn_in_field.get_value()
    sample_size = resume_pipeline.sample_size_field.get_value()
    trigger_downstream = resume_pipeline.trigger_field.get_value()
    params_str = pprint.pformat({f.key: f.get_value() for f in resume_pipeline.fields}, indent=2)

    # Decode combined app + model name from user input.
    app_name, region_name, baserun_id, base_commit = baserun.split("/")
    job_name = f"resume-{app_name}-{region_name}-{build_number}"

    logger.info("Resuming calibration job %s with params:\n%s\n", job_name, params_str)
    run_id = aws.resume_calibration(
        job=job_name,
        baserun=baserun,
        chains=chains,
        runtime=runtime,
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
            msg=f"Triggered by calibration {app_name} {region_name} (build {build_number})",
            env={"SKIP_INPUT": "true"},
            meta={
                fp.run_id_field.key: run_id,
                fp.burn_in_field.key: burn_in,
                fp.sample_size_field.key: sample_size,
                fp.commit_field.key: fp.commit_field.default,
                fp.trigger_field.key: fp.trigger_field.get_option(trigger_downstream),
            },
        )


    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    logger.info("Results available at %s", get_run_url(run_id))


@buildkite_cli.command()
def full():
    """Run a full model run job in Buildkite"""
    logger.info("Gathering data for a full model run.")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    run_id = full_pipeline.run_id_field.get_value()
    burn_in = full_pipeline.burn_in_field.get_value()
    sample_size = full_pipeline.sample_size_field.get_value()
    commit = full_pipeline.commit_field.get_value()
    trigger_downstream = full_pipeline.trigger_field.get_value()
    params_str = pprint.pformat({f.key: f.get_value() for f in full_pipeline.fields}, indent=2)
    app_name, region_name, _, _ = read_run_id(run_id)
    job_name = f"{app_name}-{region_name}-{build_number}"

    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    logger.info("Running full model run job %s with params:\n%s\n", job_name, params_str)
    aws.run_full_model(
        job=job_name,
        run=run_id,
        burn_in=burn_in,
        sample=sample_size,
        commit=commit,
    )
    if not trigger_downstream:
        logger.info("Not triggering PowerBI processing.")
    else:
        logger.info("Triggering PowerBI processing.")
        pp = powerbi_pipeline

        trigger_pipeline(
            label="Trigger PowerBI processing",
            target="powerbi-processing",
            msg=f"Triggered by full model run {app_name} {region_name} (build {build_number})",
            env={"SKIP_INPUT": "true"},
            meta={
                pp.run_id_field.key: run_id,
                pp.urunid_field.key: "mle",
                pp.commit_field.key: commit,
            },
        )
    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    logger.info("Results available at %s", get_run_url(run_id))


@buildkite_cli.command()
def powerbi():
    """Run a PowerBI job in Buildkite"""
    logger.info("Gathering data for PowerBI post processing.")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    run_id = powerbi_pipeline.run_id_field.get_value()
    urunid = powerbi_pipeline.urunid_field.get_value()
    commit = powerbi_pipeline.commit_field.get_value()
    params_str = pprint.pformat({f.key: f.get_value() for f in powerbi_pipeline.fields}, indent=2)
    app_name, region_name, _, _ = read_run_id(run_id)
    job_name = f"{app_name}-{region_name}-{build_number}"

    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    logger.info("Running PowerBI post processing job %s with params:\n%s\n", job_name, params_str)
    # FIXME +++ Nasty hack to get powerbi job running temporarily during refactor
    aws.run_powerbi(job=job_name, run=run_id, urunid=urunid, commit=commit)
    logger.info("\n=====\nRun ID: %s\n=====\n", run_id)
    logger.info("Results available at %s", get_run_url(run_id))


def get_run_url(run_id: str):
    app_name, region_name, timestamp, commit = read_run_id(run_id)
    return f"http://www.autumn-data.com/app/{app_name}/region/{region_name}/run/{timestamp}-{commit}.html"


@click.group()
def trigger():
    """
    Trigger the run of run a bunch of jobs
    """


@trigger.command("europe")
def trigger_europe():
    """
    Trigger all European mixing optimization models
    """
    logger.info("Triggering all European mixing optimisation calibrations.")
    _trigger_models(Region.MIXING_OPTI_REGIONS, trigger_europe_pipeline)


@trigger.command("victoria")
def trigger_victoria():
    """
    Trigger all Victorian models
    """
    logger.info("Triggering all Victorian regional calibrations.")
    _trigger_models(Region.VICTORIA_SUBREGIONS, trigger_victoria_pipeline)


@trigger.command("philippines")
def trigger_philippines():
    """
    Trigger all Philippines models
    """
    logger.info("Triggering all Philippines regional calibrations.")
    _trigger_models(Region.PHILIPPINES_REGIONS, trigger_philippines_pipeline)


@trigger.command("malaysia")
def trigger_malaysia():
    """
    Trigger all Malaysia models
    """
    logger.info("Triggering all Malaysia regional calibrations.")
    _trigger_models(Region.MALAYSIA_REGIONS, trigger_malaysia_pipeline)


def _trigger_models(regions, p):

    logger.info("Gathering data for calibration trigger.")
    build_number = os.environ["BUILDKITE_BUILD_NUMBER"]
    chains = p.chains_field.get_value()
    runtime = p.runtime_field.get_value()
    burn_in = p.burn_in_field.get_value()
    sample_size = p.sample_size_field.get_value()
    commit = p.commit_field.get_value()
    trigger_downstream = p.trigger_field.get_value()
    params_str = pprint.pformat(
        {f.key: f.get_value() for f in p.fields},
        indent=2,
    )
    cp = calibrate_pipeline
    for region in regions:
        model = f"covid_19:{region}"
        logger.info("Triggering model calibration %s with params:\n%s\n", model, params_str)
        trigger_pipeline(
            label=f"Trigger calibration for {model}",
            target="calibration",
            msg=f"{model} calibration triggered by bulk run (build {build_number})",
            env={"SKIP_INPUT": "true"},
            meta={
                cp.model_field.key: model,
                cp.chains_field.key: chains,
                cp.commit_field.key: commit,
                cp.runtime_field.key: runtime / 3600.0,
                cp.burn_in_field.key: burn_in,
                cp.sample_size_field.key: sample_size,
                cp.trigger_field.key: cp.trigger_field.get_option(trigger_downstream),
            },
        )


buildkite_cli.add_command(trigger)
