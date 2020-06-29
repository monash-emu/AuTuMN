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

MODEL_NAME=$(python3 -c "print('-'.join('$RUN_NAME'.split('-')[:-3]))")
GIT_COMMIT=$(echo $RUN_NAME | cut -d'-' -f4 -)

log "Updating local AuTuMN repository to run the commit $GIT_COMMIT."
sudo chown -R ubuntu:ubuntu ~/code
cd ~/code
# git fetch
# git checkout $GIT_COMMIT
git pull

log "Ensuring latest requirements are installed."
. ./env/bin/activate
pip install --quiet -r requirements.txt

log "Building input database"
python3 -m apps db build

log "Downloading MCMC databases"
aws s3 cp --recursive s3://autumn-data/$RUN_NAME/data/calibration_outputs data/calibration_outputs

# Setup folders for model runs.
mkdir -p logs
mkdir -p data/full_model_runs/

# Handle script failure
function onexit {
    log "Script exited - running cleanup code"
    log "Uploading logs"
    aws s3 cp --recursive logs s3://autumn-data/$RUN_NAME/logs
}
trap onexit EXIT

PIDS=()
DB_FILES=($(find data/calibration_outputs/ -name *.db))
NUM_DBS="${#DB_FILES[@]}"
for i in $(seq 1 1 $NUM_DBS)
do
    IDX=$(($i - 1))
    DB_FILE="${DB_FILES[$IDX]}"
    DB_NUMBER=$(echo $DB_FILE | cut -d'_' -f5 - | cut -d'.' -f1 -)
    LOG_FILE=logs/full-run-${DB_NUMBER}.log
    DEST_DB=data/full_model_runs/mcmc_chain_full_run_${DB_NUMBER}.db
    touch $LOG_FILE
    log "Running full model for chain database #$DB_NUMBER $DB_FILE"
    nohup python -m apps run-mcmc $MODEL_NAME $BURN_IN $DB_FILE $DEST_DB &> $LOG_FILE &
    PIDS+=("$!")
done

log "Waiting for ${#PIDS[@]} full model runs to complete."
for PID in ${PIDS[@]}
do
    wait $PID
done
log "All full model runs completed"

log "Uploading full model run databases"
aws s3 cp --recursive data/full_model_runs s3://autumn-data/$RUN_NAME/data/full_model_runs

log "Full model runs finished for $RUN_NAME"
