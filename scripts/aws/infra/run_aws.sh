#!/bin/bash
# https://stackoverflow.com/questions/418896/how-to-redirect-output-to-a-file-and-stdout
set -e
cd ~/code
RUN_TIME=30
NUM_CHAINS=30
CALIBRATION_NAME=malaysia
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_COMMIT=$(git rev-parse HEAD)
TIMESTAMP=$(date +%s)
RUN_NAME="$CALIBRATION_NAME-$TIMESTAMP-$GIT_BRANCH-$GIT_COMMIT"

function log {
    echo "$(date "+%F %T")> $@"
}

log "Starting calibration run $RUN_NAME"
log "Updating local AuTuMN repository to run the latest code."
git pull
. ./env/bin/activate
pip install -r requirements.txt
pip install awscli # Remove when packer re-run.

log "Running $NUM_CHAINS calibration chains for $RUN_TIME seconds."
mkdir -p logs
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

log "Uploading logs"
aws s3 cp --recursive logs s3://autumn-calibrations/$RUN_NAME/logs

log "Uploading MCMC databases"
mkdir -p data/calibration_outputs
find data -name *calibration*.db -exec mv -t data/calibration_outputs/ {} +
aws s3 cp --recursive data/calibration_outputs s3://autumn-calibrations/$RUN_NAME/data/calibration_outputs

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

log "Uploading PowerBI compatible databases"
aws s3 cp --recursive data/powerbi s3://autumn-calibrations/$RUN_NAME/data/powerbi
