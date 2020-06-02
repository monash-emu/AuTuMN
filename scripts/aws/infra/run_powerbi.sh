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

cd ~/code
log "Starting post-processing for run $RUN_NAME"
log "Updating local AuTuMN repository to run the latest code."
git pull
. ./env/bin/activate
pip install -r requirements.txt


log "Dowloading full model runs for $RUN_NAME"
aws s3 cp --recursive s3://autumn-calibrations/$RUN_NAME/data/full_model_runs data/full_model_runs

log "Collating databases"
mkdir -p data/powerbi
python -m apps db collate data/full_model_runs/ data/powerbi/collated.db


mkdir -p logs

# Handle script failure
function onexit {
    log "Script exited - running cleanup code"
    log "Uploading logs"
    aws s3 cp --recursive logs s3://autumn-calibrations/$RUN_NAME/logs
}
trap onexit EXIT

log "Adding uncertainty to collated databases"
PIDS=()
UNCERTAINTY_OUTPUTS="incidence notifications infection_deathsXall prevXlateXclinical_icuXamong"
for OUTPUT in $UNCERTAINTY_OUTPUTS
do
    log "Calculating uncertainty for $OUTPUT"
    LOG_FILE=logs/uncertainty-${OUTPUT}.log
    touch $LOG_FILE
    nohup python -m apps db uncertainty $OUTPUT data/powerbi/collated.db &> $LOG_FILE &
    PIDS+=("$!")
done

log "Waiting for ${#PIDS[@]} uncertainty calculations to complete."
for PID in ${PIDS[@]}
do
    wait $PID
done
log "All uncertainty calculations completed"

log "Pruning non-MLE runs from database"
python -m apps db prune data/powerbi/collated.db data/powerbi/pruned.db

log "Converting outputs into unpivoted PowerBI format"
FINAL_DB_FILE=data/powerbi/powerbi-${RUN_NAME}.db
python -m apps db unpivot data/powerbi/pruned.db $FINAL_DB_FILE

log "Uploading PowerBI compatible database"
aws s3 cp $FINAL_DB_FILE s3://autumn-calibrations/$RUN_NAME/data/powerbi
