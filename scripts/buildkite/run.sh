#!/bin/bash
# Entrypoint to run buildkite tasks
set -e
SCRIPT_DIR=$(dirname $0)
pushd $SCRIPT_DIR
if [ ! -d "../aws/env" ]
then
    echo "Installing requirements"
    virtualenv -p python3 ../aws/env
    . ../aws/env/bin/activate
    pip3 install --quiet -r ../aws/infra/requirements.txt
fi
. ../aws/env/bin/activate
python3 -m tasks $@
popd
