#!/bin/bash
set -e
export TZ="/usr/share/zoneinfo/Australia/Melbourne"
function log {
    echo "$(date "+%F %T")> $@"
}

log "Running PowerBI post processing"

RUN_NAME=$1
if [ -z "$RUN_NAME" ]
then
    echo "Error: First argument 'run name' missing from post-processing script."
    exit 1
fi

log "Starting post-processing for run $RUN_NAME"
log "Updating local AuTuMN repository to run the latest code."
sudo chown -R ubuntu:ubuntu ~/code
cd ~/code
git pull
. ./env/bin/activate
pip install --quiet -r requirements.txt

log "Dowloading full model runs for $RUN_NAME"
aws s3 cp --recursive s3://autumn-data/$RUN_NAME/data/full_model_runs data/full_model_runs

mkdir -p logs

# Handle script failure
function onexit {
    log "Script exited - running cleanup code"
    log "Uploading logs"
    aws s3 cp --recursive logs s3://autumn-data/$RUN_NAME/logs
}
trap onexit EXIT

log "Calculating uncertainty weights for full run databases"
UNCERTAINTY_OUTPUTS="incidence notifications infection_deathsXall prevXlateXclinical_icuXamong"
DB_FILES=($(find data/full_model_runs/ -name *.db))
NUM_DBS="${#DB_FILES[@]}"
for OUTPUT in $UNCERTAINTY_OUTPUTS
do
    PIDS=()
    for i in $(seq 1 1 $NUM_DBS)
    do
        IDX=$(($i - 1))
        DB_FILE="${DB_FILES[$IDX]}"
        DB_NUMBER=$(echo $DB_FILE | cut -d'_' -f7 - | cut -d'.' -f1 -)
        LOG_FILE=logs/weights-${DB_NUMBER}-$OUTPUT.log
        log "Calculating uncertainty weights for $OUTPUT in database #$DB_NUMBER $DB_FILE"
        nohup python -m apps db uncertainty weights $OUTPUT $DB_FILE &> $LOG_FILE &
        PIDS+=("$!")
    done
    log "Waiting for ${#PIDS[@]} uncertainty weight operations for $OUTPUT to complete."
    for PID in ${PIDS[@]}
    do
        wait $PID
    done
    log "All uncertainty weight operations for $OUTPUT completed"
done
log "All uncertainty weight operations completed"


log "Pruning full run databases"
PIDS=()
DB_FILES=($(find data/full_model_runs/ -name *.db))
NUM_DBS="${#DB_FILES[@]}"
mkdir -p data/pruned
for i in $(seq 1 1 $NUM_DBS)
do
    IDX=$(($i - 1))
    DB_FILE="${DB_FILES[$IDX]}"
    DB_NUMBER=$(echo $DB_FILE | cut -d'_' -f7 - | cut -d'.' -f1 -)
    LOG_FILE=logs/prune-${DB_NUMBER}.log
    DEST_DB=data/pruned/mcmc_pruned_${DB_NUMBER}.db
    touch $LOG_FILE
    log "Pruning  chain database #$DB_NUMBER $DB_FILE"
    nohup python -m apps db prune $DB_FILE $DEST_DB &> $LOG_FILE &
    PIDS+=("$!")
done

log "Waiting for ${#PIDS[@]} pruning operations to complete."
for PID in ${PIDS[@]}
do
    wait $PID
done
log "All pruning operations completed"

log "Collating databases"
mkdir -p data/powerbi
python -m apps db collate data/pruned/ data/powerbi/collated.db

log "Adding uncertainty to collated databases"
python -m apps db uncertainty quantiles data/powerbi/collated.db

log "Pruning non-MLE runs from database"
python -m apps db prune data/powerbi/collated.db data/powerbi/collated-pruned.db

log "Converting outputs into unpivoted PowerBI format"
FINAL_DB_FILENAME=powerbi-${RUN_NAME}.db
FINAL_DB_FILE=data/powerbi/$FINAL_DB_FILENAME
python -m apps db unpivot data/powerbi/collated-pruned.db $FINAL_DB_FILE

log "Uploading PowerBI compatible database"
aws s3 cp --acl public-read $FINAL_DB_FILE s3://autumn-data/$RUN_NAME/data/powerbi/$FINAL_DB_FILENAME

log "PowerBI processing complete"