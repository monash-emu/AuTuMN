#!/bin/bash
# Entrypoint to run AWS tasks
# IMPORTANT: This is used by the Buildkite server.
set -e
SCRIPT_DIR=$(dirname $0)
cd  $SCRIPT_DIR
cd ..
if [ ! -d "env" ]
then
    echo ">>> Creating virtualenv for Autumn scripts"
    time virtualenv --quiet -p python3 env
    . env/bin/activate
    echo ">>> Installing infra deps for Autumn scripts"
    time pip3 install --quiet -r remote/requirements.txt

fi
. env/bin/activate

python3 -m remote aws $@
