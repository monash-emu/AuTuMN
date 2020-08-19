#!/bin/bash
# Entrypoint to run tasks
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

echo ">>> Start running Autumn scripts"

time python3 -m remote $@

echo ">>> Finished running Autumn scripts"
