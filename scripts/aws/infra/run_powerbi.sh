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

log "Adding uncertainty to collated databases"
python -m apps db uncertainty data/powerbi/collated.db

log "Pruning non-MLE runs from database"
python -m apps db prune data/powerbi/collated.db data/powerbi/pruned.db

log "Converting outputs into unpivoted PowerBI format"
python -m apps db unpivot data/powerbi/pruned.db data/powerbi/powerbi-${RUN_NAME}.db

log "Uploading PowerBI compatible database"
aws s3 cp --recursive data/powerbi s3://autumn-calibrations/$RUN_NAME/data/powerbi

