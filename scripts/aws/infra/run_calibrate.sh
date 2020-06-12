#!/bin/bash
set -e
export TZ="/usr/share/zoneinfo/Australia/Melbourne"
function log {
    echo -e "\n$(date "+%F %T")> $@"
}
log "Starting calibration"

CALIBRATION_NAME=$1
NUM_CHAINS=$2
RUN_TIME=$3
if [ -z "$CALIBRATION_NAME" ]
then
    echo "Error: First argument 'calibration name' missing from calibration script."
    exit 1
fi
if [ -z "$NUM_CHAINS" ]
then
    echo "Error: Second argument 'number of chains' missing from calibration script."
    exit 1
fi
if [ -z "$RUN_TIME" ]
then
    echo "Error: Third argument 'run time (seconds)' missing from calibration script."
    exit 1
fi

log "Updating local AuTuMN repository to run the latest code."
sudo chown -R ubuntu:ubuntu ~/code
cd ~/code
git pull

log "Ensuring latest requirements are installed."
. ./env/bin/activate
pip install --quiet -r requirements.txt

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_COMMIT=$(git rev-parse HEAD)
TIMESTAMP=$(date +%s)
RUN_NAME="$CALIBRATION_NAME-$TIMESTAMP-$GIT_BRANCH-$GIT_COMMIT"

# Handle script failure
mkdir -p logs
function onexit {
    log "Script exited - running cleanup code"
    log "Uploading logs"
    aws s3 cp --recursive logs s3://autumn-calibrations/$RUN_NAME/logs
}
trap onexit EXIT

log "Starting calibration run $RUN_NAME"
log "Running $NUM_CHAINS calibration chains for $RUN_TIME seconds."
pids=()
for i in $(seq 1 1 $NUM_CHAINS)
do
    log "Starting chain $i"
    touch logs/run-$i.log
    nohup python3 -m apps calibrate $CALIBRATION_NAME $RUN_TIME $i &> logs/run-$i.log &
    pids+=("$!")
done

log "Waiting for ${#pids[@]} calibration chains to complete their $RUN_TIME second runs."
for pid in ${pids[@]}
do
    wait $pid
done
log "All chains completed"

log "Uploading MCMC databases"
mkdir -p data/calibration_outputs
find data -name *calibration*.db -exec mv -t data/calibration_outputs/ {} +
aws s3 cp --recursive data/calibration_outputs s3://autumn-calibrations/$RUN_NAME/data/calibration_outputs

log "Calibration finished for $RUN_NAME"
