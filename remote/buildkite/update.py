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
