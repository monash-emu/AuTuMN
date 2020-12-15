#!/bin/bash
SCRIPT_DIR=$(dirname $0)
DOCS_DIR=`realpath $SCRIPT_DIR/../../docs`
pushd $DOCS_DIR
    aws s3 cp --acl public-read --recursive _build/html s3://summerepi.com
popd
