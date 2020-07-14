"""
Runs AuTuMN tasks, which are large jobs to be run on remote servers.
These tasks are orchestrated using Luigi.

This module requires AWS access to run.

You can access this script from your CLI by running the Luigi CLI:
https://luigi.readthedocs.io/en/stable/running_luigi.html

export LUIGI_CONFIG_PATH=tasks/luigi.cfg

# Run a calibration
python3 -m luigi \
    --module tasks \
    RunCalibrate \
    --run-id test \
    --num-chains 2 \
    --CalibrationChainTask-model-name malaysia \
    --CalibrationChainTask-runtime 30 \
    --local-scheduler \
    --logging-conf-file tasks/luigi-logging.ini

# Run full models
python3 -m luigi \
    --module tasks \
    RunFullModels \
    --run-id test \
    --FullModelRunTask-burn-in 0 \
    --FullModelRunTask-model-name malaysia \
    --local-scheduler \
    --logging-conf-file tasks/luigi-logging.ini

# Run PowerBI processing
python3 -m luigi \
    --module tasks \
    RunPowerBI \
    --run-id test \
    --local-scheduler \
    --logging-conf-file tasks/luigi-logging.ini

"""
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


from .settings import BASE_DIR
from .calibrate import RunCalibrate
from .full_model_run import RunFullModels
from .powerbi import RunPowerBI

os.makedirs(BASE_DIR, exist_ok=True)
