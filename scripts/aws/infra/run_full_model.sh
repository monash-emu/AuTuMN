#!/bin/bash
set -e
function log {
    echo "$(date "+%F %T")> $@"
} 
log "Running full model"
MODEL_NAME=$1
RUN_NAME=$2
BURN_IN=$3
if [ -z "$MODEL_NAME" ]
then
    echo "Error: First argument 'model name' missing from model runner script."
    exit 1
fi
if [ -z "$RUN_NAME" ]
then
    echo "Error: Second argument 'run name' missing from model runner script."
    exit 1
fi
if [ -z "$BURN_IN" ]
then
    echo "Error: Second argument 'burn in value' missing from model runner script."
    exit 1
fi
log "Running full model $MODEL_NAME from $RUN_NAME with burn-in $BURN_IN"





# python -m apps run_mcmc $MODEL_NAME $SOURCE_DB

# python -m apps run-mcmc malaysia /home/matt/code/monash/autumn/data/covid_malaysia/calibration-covid_malaysia-c48be7d3-01-06-2020/outputs_calibration_chain_0.db foo.secret.db