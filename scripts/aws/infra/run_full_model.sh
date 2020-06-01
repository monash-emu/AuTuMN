#!/bin/bash
set -e
export TZ="/usr/share/zoneinfo/Australia/Melbourne"
function log {
    echo -e "\n$(date "+%F %T")> $@"
}
log "Running full model"
RUN_NAME=$1
BURN_IN=$2
if [ -z "$RUN_NAME" ]
then
    echo "Error: First argument 'run name' missing from model runner script."
    exit 1
fi
if [ -z "$BURN_IN" ]
then
    echo "Error: Second argument 'burn in value' missing from model runner script."
    exit 1
fi
log "Running full model for $RUN_NAME with burn-in $BURN_IN"

MODEL_NAME=$(echo $RUN_NAME | cut -d'-' -f1 -)
GIT_COMMIT=$(echo $RUN_NAME | cut -d'-' -f4 -)

cd ~/code
log "Updating local AuTuMN repository to run the commit $GIT_COMMIT."
git fetch
git checkout $GIT_COMMIT

log "Ensuring latest requirements are installed."
. ./env/bin/activate
pip install -r requirements.txt
pip install awscli # Remove when packer re-run.

log "Downloading MCMC databases"
aws s3 cp --recursive s3://autumn-calibrations/$RUN_NAME/data/calibration_outputs data/calibration_outputs

mkdir -p logs
mkdir -p data/full_model_runs/
DB_FILES=($(find data/calibration_outputs/ -name *.db))
PIDS=()
NUM_DBS="${#DB_FILES[@]}"
for i in $(seq 1 1 $NUM_DBS)
do
    idx=$(($i - 1))
    DB_FILE="${DB_FILES[$idx]}"
    log "Converting chain database #$i $DB_FILE"
    touch logs/full-run-$i.log
    DB_NUMBER=$(echo $DB_FILE | cut -d'_' -f5 - | cut -d'.' -f1 -)
    nohup python -m apps run-mcmc $MODEL_NAME $BURN_IN $DB_FILE data/full_model_runs/mcmc_chain_full_run_${DB_NUMBER}.db &> logs/full-run-$i.log &
    PIDS+=("$!")
done

log "Waiting for ${#PIDS[@]} full model runs to complete."
for PID in ${PIDS[@]}
do
    wait $PID
done
log "All full model runs completed"

log "Uploading logs"
aws s3 cp --recursive logs s3://autumn-calibrations/$RUN_NAME/logs

log "Uploading full model run databases"
aws s3 cp --recursive data/full_model_runs s3://autumn-calibrations/$RUN_NAME/data/full_model_runs

log "Full model runs finished for $RUN_NAME"
