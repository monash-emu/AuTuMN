#!/bin/bash
# Full model run script to be run in Buildkite
set -e
function log {
    echo -e "\n>>> $@\n"
}
log "Starting a full model run."
BURN_IN_DEFAULT=500
if [[ -z "$RUN_NAME" ]]
then
    log "Using user-supplied run name."
    BURN_IN_OPTION=$(buildkite-agent meta-data get full-burn-in)
    RUN_NAME=$(buildkite-agent meta-data get run-name)
    TRIGGER_DOWNSTREAM=$(buildkite-agent meta-data get trigger-downstream)
    BURN_IN=${BURN_IN_OPTION:-$BURN_IN_DEFAULT}
    if [[ -z "$RUN_NAME" ]]
    then
        log "No run name found"
        exit 1
    fi
else
    log "Found run name from envar: $RUN_NAME"
    TRIGGER_DOWNSTREAM=yes
    BURN_IN=$BURN_IN_DEFAULT
fi
MODEL_NAME=$(echo $RUN_NAME | cut -d'-' -f1)
JOB_NAME=$MODEL_NAME-$BUILDKITE_BUILD_NUMBER

log "Running full model for $MODEL_NAME with burn in $BURN_IN"

# Run the full models
scripts/aws/run.sh run full \
    $JOB_NAME \
    $RUN_NAME \
    $BURN_IN  | tee outputs.log

# Check output log success.
MATCH="Full model runs finished"
SUCCESS_LOG=$(sed -n "/$MATCH/p" outputs.log)

if [[ -z "$SUCCESS_LOG" ]]
then
    log "Failure to do full model run for $MODEL_NAME ($RUN_NAME)."
    exit 1
fi
log "Successfully completed full model runs for $MODEL_NAME ($RUN_NAME)."

if [ "$TRIGGER_DOWNSTREAM" == "yes" ]
then
    log "Triggering PowerBI processing."
    PIPELINE_YAML="
steps:
  - label: Trigger full model run
    trigger: powerbi-processing
    async: true
    build:
      message: Triggered by full model run ${BUILDKITE_BUILD_NUMBER}
      commit: ${BUILDKITE_COMMIT}
      branch: ${BUILDKITE_BRANCH}
      env:
        RUN_NAME: ${RUN_NAME}
"
    echo "$PIPELINE_YAML"  | buildkite-agent pipeline upload
else
    log "Not triggering PowerBI processing."
fi
