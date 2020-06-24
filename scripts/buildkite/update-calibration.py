#!/usr/bin/env python
"""
Updates the calibration pipeline in GitHub so that it includes all the latest COVID models.
"""
import os, sys

import yaml

parent_dir = os.path.abspath("../..")
sys.path.insert(0, parent_dir)

from apps.covid_19 import app

PIPELINE_PATH = "pipelines/calibrate.yml"


calibration_options = [
    {"label": rn.replace("-", " ").title(), "value": rn,} for rn in app.region_names
]

with open(PIPELINE_PATH, "r") as f:
    pipeline = yaml.safe_load(f)

pipeline["steps"][0]["fields"][0]["options"] = calibration_options

with open(PIPELINE_PATH, "w") as f:
    yaml.dump(pipeline, f)
