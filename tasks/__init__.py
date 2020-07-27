"""
Runs AuTuMN tasks, which are large jobs to be run on remote servers.
These tasks are orchestrated using Luigi.

This module requires AWS access to run.

You can access this script from your CLI by running the Luigi CLI:
https://luigi.readthedocs.io/en/stable/running_luigi.html

rm -rf data/outputs/remote data/outputs/calibrate
aws --profile autumn s3 rm --recursive s3://autumn-data/manila-111111111-aaaaaaa

./scripts/website/deploy.sh
http://www.autumn-data.com/model/manila/run/manila-111111111-aaaaaaa.html

export LUIGI_CONFIG_PATH=tasks/luigi.cfg

# Run a calibration
python3 -m luigi \
    --module tasks \
    RunCalibrate \
    --run-id manila-111111111-aaaaaaa \
    --num-chains 2 \
    --CalibrationChainTask-model-name manila \
    --CalibrationChainTask-runtime 60 \
    --local-scheduler \
    --workers 4 \
    --logging-conf-file tasks/luigi-logging.ini

# Run full models
python3 -m luigi \
    --module tasks \
    RunFullModels \
    --run-id manila-111111111-aaaaaaa \
    --FullModelRunTask-burn-in 0 \
    --FullModelRunTask-model-name manila \
    --local-scheduler \
    --workers 6 \
    --logging-conf-file tasks/luigi-logging.ini

# Run PowerBI processing
python3 -m luigi \
    --module tasks \
    RunPowerBI \
    --run-id manila-111111111-aaaaaaa \
    --local-scheduler \
    --workers 6 \
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
