#!/bin/bash
# Calibration script to be run in Buildkite
set -e
TRIGGER_DOWNSTREAM=$(buildkite-agent meta-data get trigger-downstream)
MODEL_NAME=$(buildkite-agent meta-data get model-name)
NUM_CHAINS=$(buildkite-agent meta-data get mcmc-num-chains)
RUN_TIME_HOURS=$(buildkite-agent meta-data get mcmc-runtime)
RUN_TIME=$(($RUN_TIME_HOURS * 3600))
JOB_NAME=$MODEL_NAME-$BUILDKITE_BUILD_NUMBER

function log {
    echo -e "\n>>> $@\n"
}

log "Running calbration for $MODEL_NAME with $NUM_CHAINS chains for $RUN_TIME_HOURS hours ($RUN_TIME seconds)"

# FIXME: use an exit code to show success/failure
# FIXME: write run name to a tempfile rather than parse the stdout
# Run a calibration
scripts/aws/run.sh run calibrate \
    $JOB_NAME \
    $MODEL_NAME \
    $NUM_CHAINS \
    $RUN_TIME | tee calibration.log

# Check output log for run name.
MATCH="Calibration finished for"
RUN_NAME=$(sed -n "/$MATCH/p" calibration.log | cut -d' ' -f6)

if [[ -z "$RUN_NAME" ]]
then
    log "Run for $MODEL_NAME failed."
    exit 1
fi
log "Run for $RUN_NAME suceeded."


if [ "$TRIGGER_DOWNSTREAM" == "yes" ]
then
    log "Triggering full model run."
    PIPELINE_YAML="
steps:
  - label: Trigger full model run
    trigger: full-model-run
    async: true
    build:
      message: Triggered by calibration ${BUILDKITE_BUILD_NUMBER}
      commit: ${BUILDKITE_COMMIT}
      branch: ${BUILDKITE_BRANCH}
      env:
        RUN_NAME: ${RUN_NAME}
"
    echo "$PIPELINE_YAML"  | buildkite-agent pipeline upload
else
    log "Not triggering full model run."
fi
