#!/bin/bash
# Requires watchdog
# 
#   pip install watchdog[watchmedo]
# 
SCRIPT_DIR=$(dirname $0)
ROOT_DIR=`realpath $SCRIPT_DIR/../..`
pushd $ROOT_DIR
    watchmedo shell-command \
        --command="./scripts/docs/build.sh" \
        --patterns="*.py;*.md;*.rst" \
        --ignore-patterns="./data/*;./env/*;./docs/_build/*" \
        --recursive --wait --drop \
        .
popd
