#!/bin/bash
#!/bin/bash
# Full model run script to be run in Buildkite
set -e
function log {
    echo -e "\n>>> $@\n"
}
log "Starting PowerBI post processing."
if [[ -z "$RUN_NAME" ]]
then
    log "Using user-supplied run name."
    RUN_NAME=$(buildkite-agent meta-data get run-name)
    if [[ -z "$RUN_NAME" ]]
    then
        log "No run name found"
        exit 1
    fi
else
    log "Found run name from envar: $RUN_NAME"
fi
MODEL_NAME=$(echo $RUN_NAME | cut -d'-' -f1)
JOB_NAME=$MODEL_NAME-$BUILDKITE_BUILD_NUMBER

# Run PowerBI post processing
scripts/aws/run.sh run powerbi \
    $JOB_NAME \
    $RUN_NAME | tee outputs.log

# Check output log success.
MATCH="Uploading PowerBI compatible database"
SUCCESS_LOG=$(sed -n "/$MATCH/p" outputs.log)
if [[ -z "$SUCCESS_LOG" ]]
then
    log "Failure to do PowerBI processing for $MODEL_NAME ($RUN_NAME)."
    exit 1
fi
log "Successfully completed PowerBI processing for $MODEL_NAME ($RUN_NAME)."
