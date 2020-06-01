#!/bin/bash
set -e
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
pip install awscli # Remove when packer re-run.

log "Converting outputs to PowerBI format"
mkdir -p data/powerbi
chain_dbs=($(find data/calibration_outputs/ -name *.db))
pids=()
num_dbs="${#chain_dbs[@]}"
for i in $(seq 1 1 $num_dbs)
do
    idx=$(($i - 1))
    chain_db="${chain_dbs[$idx]}"
    log "Converting chain database #$i $chain_db"
    touch logs/powerbi-convert-$i.log
    nohup python -m apps db powerbi $chain_db data/powerbi/mcmc_chain_powerbi_${i}.db &> logs/powerbi-convert-$i.log &
    pids+=("$!")
done

log "Waiting for ${#pids[@]} database conversions to complete."
for pid in ${pids[@]}
do
    wait $pid
done
log "All database conversions completed"

log "Uploading logs"
aws s3 cp --recursive logs s3://autumn-calibrations/$RUN_NAME/logs

#python -m apps db unpivot $SOURCE_DB $TARGET_DB
#python -m apps db uncertainty $SOURCE_DB $TARGET_DB [$DERIVED_OUTPUTS]
#python -m apps db prune $SOURCE_DB $TARGET_DB $FINAL_SIZE

# TODO: Collate database
# TODO: Add uncertainty

# log "Uploading PowerBI compatible database"
# aws s3 cp --recursive data/powerbi s3://autumn-calibrations/$RUN_NAME/data/powerbi
