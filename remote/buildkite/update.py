import os
import yaml


PIPELINE_PATH = "scripts/buildkite/pipelines/calibrate.yml"


def update_pipelines():
    """
    Updates the calibration pipeline in GitHub so that it includes all the latest COVID models.
    """
    from apps.covid_19 import app

    calibration_options = [
        {"label": rn.replace("-", " ").title(), "value": rn} for rn in app.region_names
    ]

    with open(PIPELINE_PATH, "r") as f:
        pipeline = yaml.safe_load(f)

    pipeline["steps"][0]["fields"][0]["options"] = calibration_options

    with open(PIPELINE_PATH, "w") as f:
        yaml.dump(pipeline, f)


def update_pipelines():
    for p in PIPELINES:
        p.save()


PIPELINES = [
    Pipeline(path="scripts/buildkite/pipelines/calibrate.yml"),
    Pipeline(path="scripts/buildkite/pipelines/run-full.yml"),
    Pipeline(path="scripts/buildkite/pipelines/powerbi.yml"),
    Pipeline(path="scripts/buildkite/pipelines/calibrate-victoria.yml"),
    Pipeline(path="scripts/buildkite/pipelines/calibrate-philippines.yml"),
]

