"""
Runs AuTuMN tasks, which are large jobs to be run on remote servers.
These tasks are orchestrated using Luigi.

You can access this script from your CLI by running the Luigi CLI:
https://luigi.readthedocs.io/en/stable/running_luigi.html

# Run a calibration
python3 -m luigi \
    --module tasks \
    RunCalibrate \
    --run-id test \
    --num-chains 2 \
    --CalibrationChainTask-model-name malaysia \
    --CalibrationChainTask-runtime 12 \
    --local-scheduler \
    --logging-conf-file tasks/luigi-logging.yml


"""
import os

import sentry_sdk

from .calibrate import RunCalibrate

# Setup Sentry error reporting - https://sentry.io/welcome/
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(SENTRY_DSN)
